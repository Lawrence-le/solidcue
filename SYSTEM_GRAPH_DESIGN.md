# System Graph Design — Agent Creation Flow

Status: **Implemented** (steps 1–7 complete — Option A interrupt landed)
Source of truth for the `graph_system` upgrade and the new `graph_definition` graph.

---

## 1. Goal

`graph_system` is the **workspace orchestrator** — it runs when there is no
`agent_key` yet (bootstrap, setup, agent creation). Today it only classifies
intent and returns a message; the `create_agent` path is a dead-end.

This design makes `graph_system` actually **create a complete, runnable agent on
disk**:

```
solidcue/agents/<agent_key>/
├── <agent_key>.yaml   ← graph_system writes (config + env keys)
├── PERSONA.md         ← graph_definition writes
├── SKILL.md           ← graph_definition writes
└── TOOLS.md           ← graph_definition writes
```

The definition files contain **real, LLM-generated content grounded in skill
contracts** — not the empty templates `create_agent()` writes today.

**Success condition:** after one create_agent pass, the agent folder is complete,
the YAML validates against `AgentConfig`, and the agent is immediately runnable
via `graph_router` / `graph_agent`.

---

## 2. Graph inventory — 4 graphs total

| # | Graph | Status | Role |
|---|---|---|---|
| 1 | `graph_agent` | exists | Executes one agent's task |
| 2 | `graph_router` | exists | Routes user intent → dispatches agents |
| 3 | `graph_system` | **upgrade** | Workspace orchestrator; drives agent creation |
| 4 | `graph_definition` | **new** | Reusable writer for an agent's definition files |

`graph_system` does **not** write the MD files itself. It orchestrates: it calls
`graph_definition` three times (persona / skill / tools), then writes the YAML
config itself.

### Who calls whom

`graph_definition` is an **internal subgraph**. It is never invoked directly by
the user or the router — **only `graph_system` calls it**:

```
user → graph_router / graph_system (entry)
          └─ graph_system orchestrates create_agent
               ├─ generate_persona_node ──calls──▶ build_persona_graph() ┐
               ├─ generate_skill_node   ──calls──▶ build_skill_graph()   ├─ graph_definition
               └─ generate_tools_node   ──calls──▶ build_tools_graph()   ┘
          └─ graph_system write_config (YAML) → verify → final_output
```

This is why `graph_definition` is **not** registered in `langgraph.json` — nothing
external reaches it.

### Wiring style (REQUIRED — do not use the alternative)

LangGraph allows two ways to run a subgraph from a parent. **Use style 1**, to
match how `graph_router` already calls `graph_agent`:

| Style | How | Use? |
|---|---|---|
| **1. Node invokes compiled subgraph** | `generate_*_node` calls `build_persona_graph()` then `.astream(...)`, collects the result into state | ✅ **Use this** — consistent with [`execute_plan_node._get_agent_graph`](solidcue/core/graph_router/nodes/execute_plan_node.py:36) |
| 2. Subgraph added as a node | `graph.add_node("persona", build_persona_graph())` | ❌ Do not use — requires parent/child state keys to line up and diverges from the existing pattern |

Under style 1 each `generate_*_node` owns the input it passes
(`agent_key`, `agent_spec`, `definition_target`) and reads back
`definition_content` / `definition_path` from the subgraph's final state, appending
to `artifacts`. Cache the compiled subgraph per target (same as `_get_agent_graph`).

### Router → graph_system wiring (create_agent from chat)

The router reaches `graph_system` for the `create_agent` intent, so "create an
agent" typed into normal chat runs the real flow:

```
user → graph_router → intent_router (classifies create_agent, seeds system_intent)
          └─ create_agent_system  ◀── graph_system embedded as a SUBGRAPH NODE
               (initialize → intent → collect_spec → … → verify → final_output)
          └─ final_output
```

**This case uses style 2 (subgraph added as a node) — the opposite of
`graph_definition` above — and that is deliberate.** `graph_system` *interrupts*
(the create-agent form); `graph_definition` does not. Style 2 lets the child
interrupt propagate to the parent run natively and lets `Command(resume=...)` flow
back down. Style 1 (`.astream()` inside a node) would run the subgraph to
completion and **swallow the interrupt**. Rule of thumb:

| Subgraph interrupts? | Style |
|---|---|
| No (`graph_definition`) | Style 1 — node invokes `.astream()` |
| Yes (`graph_system`) | Style 2 — `graph.add_node(name, compiled_subgraph)` |

Requirements that make style 2 work here:
- **Shared state channels.** `RouterState → SystemState → AgentState` inheritance
  chain, so the create-agent channels (`agent_spec`, `artifacts`,
  `created_agent_key`, `system_intent`, …) exist on the router's state and
  round-trip across the boundary.
- **No inner checkpointer / no `with_config`.** `build_system_subgraph()` compiles
  bare so the parent owns checkpointing and the interrupt propagates.
- **Upstream intent honored.** `intent_router_node` sets `system_intent =
  "create_agent"`; `graph_system.intent_node` short-circuits on it instead of
  re-classifying (which would misroute to `select_agent` when agents already
  exist).

`handoff_node` is now only the legacy no-plan fallback; `create_agent` no longer
routes to it.

---

## 3. `graph_definition` (new, reusable)

One parameterized graph, exposed via three named factories. Avoids triplicate
code while still giving three independently-callable graphs (so a future
"rewrite my SKILL.md" can call `build_skill_graph()` directly from the router).

### Directory

```
solidcue/core/graph_definition/
├── builder.py
│     build_persona_graph(), build_skill_graph(), build_tools_graph()
│     → all delegate to build_definition_graph(definition_target)
├── state/schema.py        DefinitionState
└── nodes/
    ├── load_contract_node.py   resolve + load create-<target>.md contract
    ├── generate_node.py        workspace provider → content per contract + spec
    └── write_node.py           save MD via loader (overwrite-aware)
```

### Topology

```
load_contract → generate → write → END
```

### `DefinitionState`

| Field | Type | Notes |
|---|---|---|
| `definition_target` | `"persona" \| "skill" \| "tools"` | injected by the factory |
| `agent_key` | `str` | target agent folder |
| `agent_spec` | `dict` | name / key / description / role / tools |
| `contract_skill` | `str` | loaded `create-<target>.md` text |
| `definition_content` | `str` | generated MD |
| `definition_path` | `str` | written path |
| `overwrite` | `bool` | allow regeneration over an existing file |

### Provider

`graph_definition` has no `agent_key` provider of its own — it uses the
**workspace router provider** (`_PROFILE_ROUTER_PROVIDER`, the user-profile
router provider). The provider configured *for the new agent* is part of the
spec written into the YAML; it is not the generator.

---

## 4. `graph_system` (upgraded)

### Topology

```
initialize → intent ──┬─ create_agent ─→ collect_spec ─→ [fan-out, parallel]
                      │                                    ├─ persona  (graph_definition)
                      │                                    ├─ skill    (graph_definition)
                      │                                    └─ tools    (graph_definition)
                      │                                         │ [fan-in]
                      │                                         ▼
                      │                                    write_config → verify → final_output
                      ├─ setup_provider ─→ final_output   (skill-guided, as today)
                      ├─ import_agent   ─→ final_output
                      ├─ select_agent   ─→ final_output
                      ├─ repair_config  ─→ final_output
                      └─ bootstrap      ─→ final_output
```

Key changes vs. today:
- `_route_after_intent` (currently always → `final_output`) becomes a
  **conditional router on `system_intent`**.
- The three definition subgraphs **fan out in parallel** (each is independent
  given the spec + its own contract), then fan in at `write_config`.

### New nodes

| Node | Responsibility |
|---|---|
| `collect_spec` | Validate `agent_spec`; if required fields missing → Option A interrupt / Option B error |
| `generate_persona` / `generate_skill` / `generate_tools` | Invoke the matching `graph_definition` subgraph (pattern: `execute_plan_node._get_agent_graph`) |
| `write_config` | Build provider configs + env keys; write `<agent_key>.yaml` only (MD already written) |
| `verify` | Confirm folder is complete: YAML loads, three MD files exist |

### `SystemState` additions

| Field | Type | Notes |
|---|---|---|
| `agent_spec` | `dict` | `CreateAgentInput`-shaped payload |
| `artifacts` | `Annotated[list[dict], operator.add]` | one per writer: `{target, path, content}` |
| `created_agent_key` | `str` | |
| `created_config_path` | `str` | |

### `graph_system/builder.py` layout

Node registration and edges (keep the existing checkpointer / recursion-limit
helpers as-is):

```python
def _compile_graph(checkpointer, *, session_id=None):
    graph = StateGraph(SystemState)

    # --- nodes ---
    graph.add_node("initialize",       initialize_node)        # exists
    graph.add_node("intent",           intent_node)            # exists
    graph.add_node("collect_spec",     collect_spec_node)      # new
    graph.add_node("generate_persona", generate_persona_node)  # new → graph_definition
    graph.add_node("generate_skill",   generate_skill_node)    # new → graph_definition
    graph.add_node("generate_tools",   generate_tools_node)    # new → graph_definition
    graph.add_node("write_config",     write_config_node)      # new
    graph.add_node("verify",           verify_node)            # new
    graph.add_node("final_output",     final_output_node)      # exists

    graph.set_entry_point("initialize")

    # --- edges ---
    graph.add_edge("initialize", "intent")
    graph.add_conditional_edges("intent", _route_after_intent)   # NEW conditional

    # create_agent branch: collect → fan-out (parallel) → fan-in → write → verify
    graph.add_edge("collect_spec", "generate_persona")
    graph.add_edge("collect_spec", "generate_skill")
    graph.add_edge("collect_spec", "generate_tools")
    graph.add_edge("generate_persona", "write_config")
    graph.add_edge("generate_skill",   "write_config")
    graph.add_edge("generate_tools",   "write_config")
    # write_config runs once after all three fan-in branches complete
    graph.add_edge("write_config", "verify")
    graph.add_edge("verify", "final_output")

    graph.add_edge("final_output", END)

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled.with_config({"recursion_limit": _resolve_recursion_limit()})
```

Routing function — replaces the current always-`final_output` stub:

```python
def _route_after_intent(state: SystemState):
    if state.get("system_intent") == "create_agent":
        return "collect_spec"
    return "final_output"   # setup_provider / import_agent / select_agent /
                            # repair_config / bootstrap stay skill-guided one-shot
```

Notes for the implementer:
- The three `generate_*` edges fanning into `write_config` make LangGraph wait
  for all three branches before running `write_config` once (barrier join).
- If `collect_spec` cannot proceed (missing required fields, Option B), it sets
  `system_next` / `final_response` and routes straight to `final_output` — add a
  conditional edge out of `collect_spec` for that early-exit, or have it raise
  `interrupt()` under Option A.
- Add `build_for_server(config)` (see §7) — `graph_system` has none today.

### `graph_definition/builder.py` layout

```python
def build_definition_graph(definition_target: str):
    graph = StateGraph(DefinitionState)

    graph.add_node("load_contract", load_contract_node)
    graph.add_node("generate",      generate_node)
    graph.add_node("write",         write_node)

    graph.set_entry_point("load_contract")
    graph.add_edge("load_contract", "generate")
    graph.add_edge("generate", "write")
    graph.add_edge("write", END)

    compiled = graph.compile(checkpointer=None)   # subgraph: parent owns checkpointing
    # inject the target so generate/load_contract know which contract to use
    return compiled.with_config({"configurable": {"definition_target": definition_target}})

def build_persona_graph(): return build_definition_graph("persona")
def build_skill_graph():   return build_definition_graph("skill")
def build_tools_graph():   return build_definition_graph("tools")
```

The `graph_system` `generate_*` nodes cache and invoke these (same pattern as
[`execute_plan_node._get_agent_graph`](solidcue/core/graph_router/nodes/execute_plan_node.py:36)),
passing `agent_key` + `agent_spec` and collecting `definition_content` /
`definition_path` into `artifacts`.

### Node summary (all nodes, both graphs)

| Graph | Node | New? | Responsibility |
|---|---|---|---|
| system | `initialize` | exists | Load workspace context (agents, skill keys) |
| system | `intent` | exists | Classify `system_intent`; load matching skill |
| system | `collect_spec` | new | Validate `agent_spec`; gate or interrupt on missing fields |
| system | `generate_persona` | new | Invoke `build_persona_graph()` |
| system | `generate_skill` | new | Invoke `build_skill_graph()` |
| system | `generate_tools` | new | Invoke `build_tools_graph()` |
| system | `write_config` | new | Build providers + env keys; write `<agent_key>.yaml` |
| system | `verify` | new | Confirm YAML loads + 3 MD files exist |
| system | `final_output` | exists | Stable user-facing response |
| definition | `load_contract` | new | Load `create-<target>.md` into `contract_skill` |
| definition | `generate` | new | Workspace provider → `definition_content` per contract + spec |
| definition | `write` | new | Save MD via loader (overwrite-aware) |

---

## 5. Skill contracts (`solidcue/skills/`)

Contracts are the single source of truth for each file's output structure, which
keeps quality consistent across LLMs. Make them symmetric.

```
solidcue/skills/
├── create-agent.md     exists  (orchestration + config contract)
├── create-skill.md     exists  (SKILL.md contract)
├── create-persona.md   NEW     (PERSONA.md contract — mirror create-skill.md)
├── create-tools.md     NEW     (TOOLS.md contract — mirror create-skill.md)
└── user-profile.md     exists
```

- **Filenames must be hyphenated** (`create-persona.md`) — `resolve_system_skill_key()`
  normalizes `_`→`-` ([`workspace_service.py:66`](solidcue/services/workspace_service.py)),
  so an underscore filename won't resolve.
- Extend `resolve_system_skill_key()` to map `create-persona` and `create-tools`.
- `graph_definition.load_contract_node` resolves `definition_target` →
  `create-<target>.md` and loads it.

---

## 6. Tool / function layer changes

### `agent_configs/loader.py`
- `save_agent_persona` / `save_agent_skill` / `save_agent_tools` currently **raise
  `FileExistsError`** on an existing file. Add `overwrite: bool = False` (or
  `update_agent_*` variants) so regeneration works.

### `services/agent_service.py`
- **Split `create_agent`** into:
  - `write_agent_config(spec)` → env keys + YAML only (no MD writes)
  - thin `create_agent(spec)` that composes config + MD generation, so existing
    REST (`POST /agents`) and CLI paths keep working unchanged.
- `write_config_node` (graph_system) calls `write_agent_config`; the MD files
  come from `graph_definition`.

---

## 7. `langgraph.json` registration

Currently only `agent` and `router` are registered. Add `system`:

```json
{
  "graphs": {
    "agent":  "solidcue.core.graph_agent.builder:build_for_server",
    "router": "solidcue.core.graph_router.builder:build_for_server",
    "system": "solidcue.core.graph_system.builder:build_for_server"
  }
}
```

`graph_system.builder` needs a `build_for_server(config)` factory (it has none
today). `graph_definition` is invoked *inside* `graph_system`, so it does not
need its own server registration.

---

## 8. Spec / API-key collection — frontend-form-on-interrupt

Each agent carries its own provider configs: **4 roles** (decision/brain, lite,
reviewer, writer) × **5 fields** (provider_type, base_url, model, temperature,
api_key). API keys cannot be invented and must not be typed into chat (they would
be checkpointed). So `collect_spec_node` validates the **full `CreateAgentInput`**
and, when incomplete, **interrupts with a form schema** the frontend renders as a
secure create-agent form. Two paths coexist:

- **Option B — pre-supplied spec:** `POST /agents` → complete `CreateAgentInput`
  → no interrupt; validate and proceed.
- **Option A — frontend form on interrupt:** an incomplete/invalid spec pauses
  with a `collect_agent_spec` payload. The frontend renders the form (provider
  inputs, API keys as password fields), then resumes with
  `Command(resume={"agent_spec": {<full spec>}})`. The reply is merged and
  re-validated; still-invalid → `final_output` error.

### Validation

Missing/invalid fields are derived by attempting `CreateAgentInput(**agent_spec)`
and reading the `ValidationError` locations — no hand-maintained required list, so
the gate never drifts from the model. `writer_*` is optional; everything else
(incl. the three role `*_api_key`s and `selected_tools`) is required.

### Interrupt payload contract

```python
interrupt({
    "type": "collect_agent_spec",
    "agent_spec": {...},              # what we have so far (pre-fill)
    "invalid_fields": ["decision_api_key", "lite_model", ...],  # from ValidationError
    "form_schema": {
        "basic": ["name", "agent_key", "description", "selected_tools"],
        "provider_roles": ["decision", "lite", "reviewer", "writer"],   # writer optional
        "provider_fields": ["provider_type", "base_url", "model", "temperature", "api_key"],
        "secret_fields": ["api_key"],          # frontend renders as password, never echoes
        "provider_types": ["anthropic", "openai", "openrouter"],
        "available_tools": [...],              # tool_keys from the tool registry
    },
    "message": "Fill in the agent details and provider settings.",
})
```

### Secret hygiene

`write_config_node` writes the API keys to the env store, then **scrubs every
`*_api_key` field from `agent_spec`** before returning, so raw secrets do not
persist in the checkpointed graph state.

### Behavior summary

| Incoming `agent_spec` | collect_spec result |
|---|---|
| Complete & valid | proceeds to generate_* (no interrupt) |
| Missing/invalid fields | `interrupt()` with `form_schema` → resume merges + re-validates |
| Still invalid after resume | `final_output` with "required fields missing or invalid" |

Interrupt requires a checkpointer — provided by the server (`build_for_server`),
the sync `SqliteSaver` (`build_system_graph`), or `InMemorySaver` in tests.

### Frontend responsibility

The frontend owns secure key entry and the resume payload. Keys flow through
`Command(resume=...)` into state only transiently (scrubbed after `write_config`).
If per-agent keys are not needed, a future option is to reference a workspace
provider registry instead — out of scope here.

---

## 9. Build order (for the coding agent)

1. Add `skills/create-persona.md`, `skills/create-tools.md`; extend
   `resolve_system_skill_key()`.
2. Add `overwrite` support to `save_agent_*`; split `agent_service` into
   `write_agent_config` + thin `create_agent`.
3. Build `graph_definition` (state, 3 nodes, 3 factories). Unit-test each writer
   standalone (persona / skill / tools).
4. Extend `SystemState`; add `collect_spec`, the three `generate_*` nodes,
   `write_config`, `verify`.
5. Rewire `graph_system/builder.py`: conditional routing on `system_intent`,
   parallel definition branch, fan-in. Add `build_for_server`.
6. Register `system` in `langgraph.json`.
7. Option A interrupt-based collection in `collect_spec_node` (see §8). ✅ done

---

## 10. Naming decisions (locked)

- New graph is **`graph_definition`** (not `graph_artifact` — "artifact" is
  reserved; `target_artifacts_source` already uses it. Not `graph_context` —
  "context" is overloaded in LLM code).
- State target field is **`definition_target`**; content is
  **`definition_content`**; path is **`definition_path`**.
- Factories: `build_persona_graph()`, `build_skill_graph()`, `build_tools_graph()`.
