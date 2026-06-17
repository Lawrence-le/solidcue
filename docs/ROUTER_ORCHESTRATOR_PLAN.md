# Router-as-Manager (Orchestrator) Plan

## Goal

Turn the user-facing **router** into a **manager/supervisor**: it decides which agent(s)
a request needs, dispatches each agent to do its work, collects their outputs, then
**synthesizes a single final response** back to the user.

Today the router is a *receptionist*: it picks one agent and streams that agent's output
straight to the user, then exits. We want it to stay in the loop, gather worker outputs,
and speak last.

### Decisions locked in
- **Manager pattern**: router decides → workers do work → return to router → router replies.
- **Dispatch**: **sequential** first (each worker can see prior workers' output; reuses the
  existing single-agent streaming path). Evolve to per-task parallelism later.
- **Streaming UX**: **live sub-agent progress** — stream each worker's events tagged by
  `agent_key`, then stream the router's final synthesis.

### Non-goals (for this iteration)
- Parallel fan-out execution (designed for, not built).
- Agents calling each other directly (always go through the router).
- Replacing the worker agent graph's internal Phase-3 `task_plan`. That is the *worker's*
  internal decomposition; this plan adds an **outer** manager layer above it. Distinct layer.

---

## Current flow (single agent)

```
initialize → intent_router → handoff → final_output
                  │
                  └─ picks ONE target_agent_key
```

In `run_engine._stream_router_chat_events_direct`:
- `run_engine.py:1544` — if a single `target_agent_key` + `handoff_action == "route_agent"`:
- `run_engine.py:1594` — emit ONE `handoff` event.
- `run_engine.py:1603` — `stream_agent_graph_events(...)` streams that agent to the user.
- `run_engine.py:1611` — `return`. The router never speaks again.

Single-agent assumptions are baked into:
- `solidcue/core/graph_router/state/schema.py:22` — `target_agent_key: str` (scalar).
- `solidcue/core/graph_router/prompts/router_system_prompt.py:16` — "Choose **the best**".
- `solidcue/services/run_engine.py:1588` — `upsert_conversation(agent_key=…, last_thread_id=…)`
  models one active agent per turn.

---

## Target flow (manager)

```
initialize → intent_router(plan) → orchestrate(loop workers) → synthesize → final_output
                   │                       │                        │
                   │                       │ for each step:         │ router LLM call over
                   │                       │  run worker, stream    │ {user_input + all
                   │                       │  live, CAPTURE output  │  worker outputs}
                   └─ emits PLAN:          │                        │
                      [{agent_key,         └─ writes agent_results  └─ writes synthesis_draft
                        sub_task}, …]
```

The user sees: live worker progress (tagged per agent) → final synthesized answer from the router.

---

## Work breakdown

Ordered so each slice is independently testable. Slices 1–2 are the easy, self-contained
router-side change; slice 3 is the bulk of the effort.

### Slice 1 — Router emits a plan (EASY, ~1–2 hrs)

**`solidcue/core/graph_router/state/schema.py`**
- Add:
  - `plan: list[dict[str, Any]]` — each item `{agent_key, sub_task, depends_on?: list[int]}`.
  - `agent_results: Annotated[list[dict[str, Any]], operator.add]` — append-only worker
    outputs `{agent_key, sub_task, output, status}`.
  - `synthesis_draft: str` — router's combined answer (mirrors the worker graph's key name).
- Keep `target_agent_key`/`handoff` for backward compatibility + the single-agent fast path
  (a 1-item plan can still flow through the existing path during migration).

**`solidcue/core/graph_router/prompts/router_system_prompt.py`**
- For `router_intent == "task"`, emit a `plan` array instead of a single `target_agent_key`.
- Each plan item: `{ "agent_key": "...", "sub_task": "what this agent should do" }`.
- Instruction shift: "Choose the best target_agent_key" → "List every agent needed and the
  specific sub-task for each. Use one item when one agent suffices."
- Add a second prompt builder, `build_router_synthesis_prompt()`, used by the synthesis step:
  inputs are the original user message + each worker's `{agent_key, sub_task, output}`; output
  is one cohesive user-facing reply (no raw dumps, no "Agent X said…").

**`solidcue/core/graph_router/nodes/intent_router_node.py`**
- Parse `plan` from the model JSON (`intent_router_node.py:130` area).
- Fallback: if `plan` missing but `router_intent == "task"`, synthesize a 1-item plan from the
  existing `select_target_agent_key` path so nothing regresses.
- Validate every `agent_key` against `_available_agents()`; drop unknown keys, and if the plan
  becomes empty, downgrade to `clarify`.

**Tests** (`tests/test_router_intent_node.py`): a multi-agent message yields a 2+ item plan;
unknown agent keys are filtered; empty-plan → clarify; single-agent still works.

### Slice 2 — Synthesis node + graph wiring (MEDIUM, ~half day)

**New `solidcue/core/graph_router/nodes/synthesize_node.py`**
- Reads `agent_results` + `user_input`, calls the router provider with
  `build_router_synthesis_prompt(...)`, writes `synthesis_draft` (and `final_response`).
- Streams via the same custom-stream channel the router already uses (so `message_delta`
  events flow to the user — see `run_engine.py:1502` custom handler).
- Mirror the threading care in `intent_router_node.py:88` (`asyncio.to_thread` around the
  blocking provider call).

**`solidcue/core/graph_router/builder.py`**
- Add node `synthesize`.
- Routing: `intent_router` with a task plan → `orchestrate` (new) → `synthesize` → `final_output`.
- Keep the `chat`/`clarify`/`create_agent` paths going straight to `final_output` unchanged.
- Note: the actual worker dispatch happens in `run_engine` (it owns the agent sub-streams and
  the SSE feed), not inside the router graph. The `orchestrate` graph node mainly records plan
  progress; the loop that *runs* workers lives in run_engine (Slice 3). Decide during impl
  whether `orchestrate` is a real graph node or purely a run_engine concern — leaning
  run_engine, with the router graph going `intent_router → synthesize` and run_engine
  interleaving worker runs before driving the synthesis step.

**Tests**: synthesis node combines two fake `agent_results` into one reply; streams deltas.

### Slice 3 — Orchestration loop in run_engine (the BULK, ~1–1.5 days)

This rewrites `run_engine.py:1544–1611` from "stream one agent then return" to
"loop, capture, then drive synthesis".

**Replace the single-agent block with a sequential loop:**
```text
results = []
for i, step in enumerate(plan):
    agent_thread_id = create_thread_id()
    upsert_conversation(..., agent_key=step.agent_key, last_thread_id=agent_thread_id,
                        last_run_status="running")
    yield handoff event { target_agent_key: step.agent_key, agent_thread_id,
                          step_index: i, step_count: len(plan) }   # UI tags this worker
    captured = []
    async for event in stream_agent_graph_events(
            agent_key=step.agent_key,
            thread_id=agent_thread_id,
            conversation_id=resolved_conversation_id,
            user_input=_compose_subtask_input(step, results),   # prior outputs as context
            record_user_message=False):
        # tag event with step.agent_key / step_index so the UI separates workers
        yield _tag(event, step.agent_key, i)
        captured.append(event)
    results.append({ "agent_key": step.agent_key, "sub_task": step.sub_task,
                     "output": _extract_final_text(captured, agent_thread_id),
                     "status": "completed" })
# then: feed results into the router synthesis step and stream synthesis_draft to the user
```

**Capturing each worker's final output** — two viable approaches, pick during impl:
1. **Accumulate `message_delta` events** for that worker's thread from the stream (cheap,
   already flowing). `_extract_final_text` joins the deltas.
2. **`graph.aget_state` on the agent thread** after its stream ends and read the worker's
   `synthesis_draft`/`final_response` (authoritative; the worker graph already produces this
   per `AGENT_GRAPH_REDESIGN.md` `final_output` node). Preferred for fidelity.

**Driving synthesis after the loop:**
- Resume the router graph (it owns the `synthesize` node + provider config keyed by
  `router_thread_id`) with `agent_results` populated, OR call the synthesis provider inline in
  run_engine using `build_router_synthesis_prompt`. Prefer routing it through the graph so the
  checkpoint/state stays consistent and the custom-stream path is reused.
- Stream `synthesis_draft` deltas to the user as the router's `message_delta`s, then
  `append_chat_message(role="assistant", agent_key="router", content=synthesis)`.
- Finalize the checkpoint timer (`_finalize_checkpoint_timer`, as `final_output`).

**Conversation/thread model** — today singular (`run_engine.py:1588`). Options:
- Keep `upsert_conversation` per worker as the loop runs (last write wins = last worker), and
  record `agent_key="router"` at the synthesis step so the conversation's final owner is the
  router. Simplest; acceptable for sequential.
- If per-worker thread history needs to be addressable later, that's a follow-up schema change
  to `state_snapshot_service` — out of scope here.

**Tests** (`tests/test_api_streaming.py`, `tests/test_agent_service_langgraph.py`):
- 2-step plan → two `handoff` events tagged by step, two worker streams, then router
  `message_delta`s for the synthesis.
- Worker output is captured and present in the synthesis prompt (assert via a fake provider).
- A failing worker (status != completed) still lets synthesis run with partial results.

### Slice 4 — Studio UI: tagged worker progress (MEDIUM, frontend)

`studio/` consumes the SSE events. Today it renders a single handoff + one agent stream.
- Render each `handoff` event as a labeled worker lane (`agent_key`, step i/N).
- Route tagged `message_delta`s to the right worker lane; the final untagged
  (router) `message_delta`s render as the assistant's combined answer.
- Files: `studio/src/lib/api.ts` (event parsing), the chat/session view components.
- Confirm exact files against the live SSE shape once Slice 3 emits the new fields.

---

## Effort summary

| Slice | Scope | Effort |
|---|---|---|
| 1 | Router emits plan (schema, prompt, intent node) | ~1–2 hrs |
| 2 | Synthesis node + graph wiring | ~half day |
| 3 | run_engine sequential orchestration loop + capture + synthesis drive | ~1–1.5 days |
| 4 | Studio tagged worker lanes | ~half day |

**Total: ~2.5–3 days.** Slice 1 is shippable on its own (router *picks* multiple agents,
single-agent execution unchanged) and de-risks the rest.

---

## Open questions to resolve during implementation

1. **Capture method** — accumulate `message_delta`s vs `aget_state` on the worker thread for
   its `final_response`. (Leaning `aget_state` for fidelity.)
2. **Synthesis drive** — resume the router graph's `synthesize` node vs inline provider call in
   run_engine. (Leaning graph node for state/stream consistency.)
3. **`orchestrate` as a graph node vs run_engine-only** — workers run as separate graphs in
   run_engine; the router graph may not need a real `orchestrate` node. (Leaning run_engine
   owns the loop; router graph is `intent_router → synthesize`.)
4. **Conversation ownership** when N workers + a router all write in one turn. (Leaning: router
   is the final owner; per-worker thread addressability is a follow-up.)
5. **Failure policy** — one worker fails: continue with partial results (recommended) vs abort.

---

## Alignment notes

- Per CLAUDE.md domain-agnosticism: keep all of this generic — no domain literals in
  `solidcue/core` or prompts. The plan/sub-task text comes from the LLM + agent registry, not
  hardcoded.
- This sits **above** the worker graph's Phase-3 `task_plan` (`AGENT_GRAPH_REDESIGN.md:505`).
  Manager plan = which agents. Worker `task_plan` = how one agent decomposes its own work.
  They compose; don't conflate them.
```
