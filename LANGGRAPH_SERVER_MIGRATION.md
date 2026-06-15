# LangGraph Server Migration Plan

**Status:** Planned — not started
**Decided:** 2026-06-15
**Source of truth:** this doc.

## Guiding principle

**Migrating to native LangGraph means removing our non-standard / hand-coded infrastructure, not porting it.** Every hand-rolled piece that duplicates something LangGraph Server provides natively (run tracking, thread/conversation storage, message history, SSE streaming, schema migrations) should be **reduced or deleted**, not re-implemented. The goal is to **align with LangGraph's official way of working** so we maintain agent IP — not infrastructure. When in doubt: if LangGraph has a native primitive for it (threads, runs, assistants, store, checkpointer), use that primitive and drop our version.

## Why

Move the backend from **library-only LangGraph** (own FastAPI + hand-rolled in-memory run engine) to **native LangGraph Server**.

The trigger is `solidcue/services/run_engine.py` (~1,861 lines) — an in-memory reimplementation of LangGraph Server's run manager, but missing persistence and stream re-attach. The visible symptom: **"rejoin after refresh" is broken.** On refresh the SSE socket drops; events go into a throwaway `asyncio.Queue` with no replay buffer, so rejoin has nothing to re-attach to. The engine is deemed unmaintainable and bug-prone.

LangGraph Server gives us natively what we hand-built and what we're missing:

| Concern | Native LangGraph Server | What replaces |
|---|---|---|
| Run lifecycle | Persisted `Run` objects | `_ACTIVE_RUNS` / `_RUN_TASKS` in-memory dicts |
| Streaming + resume | `runs.join_stream(thread_id, run_id)` (listen-only re-attach) | hand-rolled SSE + throwaway queues |
| Chat history | `thread` (thread_id = the conversation) | `chat_history_service.py` |
| Sub-agents | subgraphs (one run) or spawned child runs (own thread) | manual `::worker::` thread tracking |

## Confirmed safe

- **Graphs/nodes do NOT break.** They are already native LangGraph `StateGraph` definitions; the server just hosts the same compiled graphs.
- **Run engine is fully decoupled** from graph logic — no node imports `run_engine`, `_RUN_QUEUES`, or `_ACTIVE_RUNS`. Deleting it touches zero nodes.

## Code impact

**Deleted (~2,900 lines of plumbing):**
- `solidcue/services/run_engine.py` (1,861)
- `solidcue/services/chat_history_service.py` (393)
- `solidcue/api/routes/state.py` (301), `chat.py` (57), most of `agents.py` (298)

**Kept (~6,000 lines — the actual IP):**
- `solidcue/agents/**`, `solidcue/core/graph_*/nodes/**`, graph builders (minus checkpointer wiring)

**Frontend:** `studio/src/lib/api.ts` hand-rolled fetch/SSE → `@langchain/langgraph-sdk` client.

---

## Three adaptation points (NOT breakage — scoped rewiring)

These are the only places graph-adjacent code changes. Each is an explicit task; do them deliberately.

### A. Checkpointer wiring — drop it from builders
The server injects its own checkpointer. Our builders currently compile with our own `SqliteSaver`/`AsyncSqliteSaver`:
- `solidcue/core/graph_router/builder.py:107` — `graph.compile(checkpointer=...)`
- `solidcue/core/graph_agent/builder.py:153` — `graph.compile(checkpointer=...)`

**Task:** expose the graph **without** our checkpointer (or expose the uncompiled graph) so the server owns persistence. Mechanical; graph definition untouched.

### B. History-reading nodes — read from state, not the side DB
Native: the **thread/graph-state IS the chat history**; nodes read messages from state, not an external service. These 4 nodes call `load_chat_history(...)` and must be rewired:
- `solidcue/core/graph_router/nodes/initialize_router_node.py`
- `solidcue/core/graph_agent/nodes/discovery_node.py`
- `solidcue/core/graph_agent/nodes/decision_node.py`
- `solidcue/core/graph_agent/nodes/planning_node.py`

**Task:** rework to pull history from graph state. This is the highest-care item — where a careless migration would introduce bugs.
*(Unrelated services `workspace_service` and `hhem_service` stay as-is.)*

### C. Re-home Langfuse tracing — RESOLVED (bake callback into compiled graph)
Langfuse is attached as LangChain callbacks **inside the file being deleted**:
- `solidcue/services/run_engine.py:176-178` and `:1489-1491` — `run_config["callbacks"] = get_langfuse_callbacks()`
- handler defined in `solidcue/observability/langfuse.py:36`

Under the server we don't own the `astream(config=...)` call, so this injection point vanishes — tracing goes dark unless re-homed.

**Verified (2026-06-15):** `langfuse>=4.7.0` for **LangChain is still callback-based** — there is NO `instrument_langchain()` / global OTel auto-instrumentation that traces LangChain/LangGraph without per-run callbacks. (Env vars: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`.) The earlier "server-runtime OTel" direction is NOT available for the LangChain path.

**Solution:** bake the callback into the compiled graph at build time — `_compile_graph(...).with_config({"callbacks": get_langfuse_callbacks()})` in both builders. Moves injection from per-run (run_engine) to compile-time, so it traces on BOTH the FastAPI and LangGraph Server paths with no per-call wiring. ~2-line change per builder. *(Confirm a single shared handler is safe across concurrent runs — expected yes.)*

---

## DB schema absorption

**No data migration. The DB has been cleared — fresh start.** There is NO legacy data to carry over and NO schema to migrate. Do **not** add any DB data-migration or schema-migration steps to this plan. The old hand-coded tables are simply abandoned; LangGraph Server creates and owns its own schema from scratch.

The table below is a **reference mapping only** (what each old column corresponded to, so we know nothing of value is lost) — it is NOT a migration instruction. Most hand-coded columns existed *because* we were tracking threads/runs/history ourselves — the exact job LangGraph Server takes over. **Per the guiding principle: drop these tables, don't port them.**

**`chat_history` table → fully absorbed (delete entirely):**
| Column | Native LangGraph |
|---|---|
| `role`, `content` | thread state (the message list) |
| `created_at` | thread message ordering |
| `conversation_id` | = thread_id |

**`conversations` table → mostly absorbed (collapses into threads + native run tracking):**
| Column | Native LangGraph |
|---|---|
| `conversation_id` | = **thread_id** (the thread *is* the conversation) |
| `last_thread_id` | redundant once conversation == thread |
| `last_run_id`, `last_run_status` | **server tracks runs natively** — query runs per thread via API; do not store. (This is the run-tracking that didn't survive restart.) |
| `created_at`, `updated_at` | thread metadata/timestamps |
| `agent_key` | ⚠️ **the one field needing a deliberate home** — see below |

**`agent_key` — the only genuinely domain-specific field.** "Which agent owns this conversation." Native homes:
- map to **`assistant_id`** (each graph/config becomes an assistant), or
- store in the thread's **metadata** bag.

**Decision required during Phase 1:** pick assistant_id vs. thread metadata for `agent_key`. Everything else in these two tables is deleted.

**Bonus removal:** the hand-rolled `ALTER TABLE … ADD COLUMN` schema migrations (`chat_history_service.py:50-54, 90-100`) go away — the server owns its own schema. No more hand-maintained migrations.

## Phased path (de-risked — resume must work in the new path before deleting the old)

### Phase 0 — Spike (✅ DONE — PASSED 2026-06-15)
- Added `langgraph-cli[inmem]>=0.4.3` + `langgraph-sdk>=0.2.9`; installed `langgraph-api==0.9.0`, `langgraph-cli==0.4.29`, `langgraph-runtime-inmem==0.29.0`.
- `langgraph.json` with two graph entries (agent factory + spike graph). Factory `build_for_server(config)` added to `graph_agent/builder.py`.
- `scripts/spike_graph.py` (5-node sleep graph) + `scripts/spike_resume.py` (baseline vs. disconnect+rejoin). *(Both scripts removed post-migration; spike entry also removed from `langgraph.json`.)*
- **Result: PASS.** baseline `[a,b,c,d,e]`; disconnect after `[a,b]`; rejoin recovered `[c,d,e]`; combined == baseline, no loss/dupes/reorder. Node `c` ran *during* the disconnect window and was replayed correctly.
- **Deceptive case confirmed:** first attempt LOST `c` — `join_stream` is a live tail by default. Resume only works with the two findings below.

#### Phase 0 findings — these are HARD REQUIREMENTS for Phase 1 (not trivia)
1. **`stream_resumable=True` is mandatory** when creating/starting a run. It tells the server to assign SSE event IDs and buffer events for replay. Without it, `join_stream` is live-only and gap events are silently lost.
2. **`last_event_id` is the resume mechanism.** Capture `chunk.id` (the SSE event ID via `StreamPart.id`) from the last received event; pass it to `join_stream(thread_id, run_id, last_event_id=...)`. Server replays everything after that ID. No dedup needed. *(This is the native equivalent of the `Last-Event-ID` pattern.)*
3. **`stream_mode` must be a single string** on langgraph-api 0.9.0 (e.g. `"updates"`), NOT a list. `"metadata"` is not a valid mode — metadata events arrive automatically.
4. In-memory mode (`langgraph dev`) IS re-attachable — `join_stream` re-attached to a run still executing in the background. (Full durability across server restart is still a Postgres+Redis / `langgraph up` property.)

> ⚠️ **CRITICAL refresh implication for Phase 1:** on a real page refresh the in-memory `last_event_id` JS variable is destroyed. The frontend MUST **persist `last_event_id` (keyed by run_id/thread_id) to storage** (e.g. sessionStorage/localStorage) as events arrive, then read it back on reload to pass into `join_stream`. If it rejoins with no `last_event_id`, it falls back to a live tail and the gap is lost again — re-introducing the exact bug. This is the make-or-break detail of the UI resume.

### ⚠️ STRUCTURAL DISCOVERY (Phase 1 investigation, 2026-06-15)
`run_engine.py` is NOT just plumbing — it contains the **multi-agent orchestration**. The router graph only *routes* (chat/clarify/task intent) and produces a `plan`; the *execution* of that plan lives in `run_engine`:
- `plan` event emitted at `run_engine:1626` (formats router's `state["plan"]`).
- `subagent`/`subagent_delta` at `run_engine:1649-1691` — run_engine iterates the plan and calls `stream_agent_graph_events()` per step. **The router graph does not execute sub-agents at all.**
- `message_delta` synthesized by run_engine from LLM token chunks (`run_engine:710,722,757,1575,1748,1765,1828`).

So `plan`/`subagent`/`message_delta` are **imperative emissions from run_engine**, not node returns or custom events. `stream_mode="updates"` alone cannot reconstruct `subagent`/`subagent_delta` — they don't exist in graph state.

**stream_mode decision:** use `["updates","messages","custom"]` (all valid enum modes — Phase 0's failure was `"metadata"`, not the list form). `messages` = token streaming (replaces `message_delta`); `custom` = `plan`/`subagent` re-emitted as dispatched events; `updates` = state diffs for structure.

### Decision: Option A (graph-native fan-out), sequenced into two waves
**Chosen 2026-06-15.** Port the fan-out into the graph rather than leaving task intents on run_engine. Rationale: (1) the runs users refresh during ARE the long multi-agent task runs — leaving them on run_engine leaves the original bug unfixed; (2) keeps run_engine in the hot path, contradicting the migration's purpose; (3) under Option A, `plan`/`subagent`/`message_delta` come back as custom events, so NO UX downgrade. De-risk via sequencing, not via the rejected Option B (phase split).

**Wave 1 — prove the pipe (chat/clarify):**
- Add **router-graph factory** (`graph_router/builder.py`) + register in `langgraph.json`. Drop manual checkpointer. *(Step 1 — DONE.)*
- Adaptation B (history write-back + read-from-state) and C (Langfuse compile-time callback). *(Backend, independent.)*
- Frontend → LangGraph SDK (`@langchain/langgraph-sdk`), applying Phase 0 findings:
  - create runs with **`stream_resumable=True`**
  - track `chunk.id`; **persist `last_event_id` to storage keyed by run_id** (survives refresh); on reload `join_stream(..., last_event_id=<persisted>)`
  - `stream_mode = ["updates","messages","custom"]`
- Cut **chat/clarify** intents over. `agent_key` home = **assistant_id** (decided).
- **Wave 1 exit:** chat/clarify resume in a real browser refresh with zero gap-event loss; run_engine still handles task intents.

**Wave 2 — port the fan-out (task intents):**
- Add an `execute_plan` node to the router graph that runs the agent graph **imperatively** inside the node via `agent_graph.astream()`, re-forwarding events to the outer stream via `get_stream_writer()`. This is NOT a wired nested-subgraph node — wiring would auto-propagate `updates`/`messages` but would suppress `custom` events, breaking `plan`/`subagent`/`subagent_delta` shapes.
- Re-emit `plan`/`subagent`/`subagent_delta`/`message_delta` as **custom events** dispatched from the node via `get_stream_writer()`.
- **Known limitation (production phase):** The checkpointer saves at node-completion boundaries only. If the LangGraph Server process restarts mid-fan-out, `execute_plan` is checkpointed as incomplete and the whole node re-runs from scratch — re-executing already-completed sub-agent steps. This is acceptable for Wave 2 (browser-refresh resume is unaffected; it operates at the SSE buffer level, not the checkpoint level). Address in production phase by either: (a) breaking each step into its own router graph node, or (b) using a LangGraph Store to persist per-step results so re-execution is idempotent.
- **Wave 2 exit:** task intents resume too; `run_engine.py` no longer in the hot path.

- FastAPI stays as companion throughout Phase 1.
- **Phase 1 exit criteria:** ALL intents (chat/clarify/task) resume in the UI with zero gap-event loss across a real page refresh; `run_engine.py` out of the hot path.

#### Phase 1 result: ✅ BOTH WAVES PASSED (2026-06-15)
- **Wave 1 (chat/clarify):** resume after real browser refresh proven, zero gap loss.
- **Wave 2 (task fan-out):** `execute_plan_node` ports the run_engine orchestration into the graph (imperative `agent_graph.astream()` forwarding inside one node + `get_stream_writer()` custom events — NOT a true nested subgraph). Gate PASS: baseline 2-step task run completes (jd_archiver → resume_builder → synthesis → final_output, 60 events); mid-fan-out browser refresh rejoins from `last_event_id`, replays the gap event (`resume_builder running`) as the first replayed event, and streams through to completion with synthesis. Resume proven for multi-agent task runs.
- **8 inner-agent nodes converted sync→async** (decision, discovery, planning, final_output, reflection, synthesis, validation_llm, validation_hhem) — a blocking sync `provider.generate()` inside the async server was starving the event loop and timing out on the streaming-only `step_plan` endpoint. All now use `timed_async_stream_generate` (joins chunks to a single string; parse logic unchanged). *Likely the task path never completed cleanly end-to-end before this migration.*

#### ⚠️ KNOWN LIMITATION (revisit in production phase)
`execute_plan` is **one atomic node** — no checkpoint BETWEEN sub-agent steps. Browser-refresh rejoin works (events buffer at the stream level via `stream_resumable=True`). But a true execution-resume (server restart mid-fan-out) re-runs the whole `execute_plan` node from scratch, re-executing completed steps. Acceptable for refresh-rejoin; address with per-step checkpointing (true subgraph) when moving to the Postgres+Redis production stack.

#### ⚠️ REQUIRED FOLLOW-UP (before production sign-off — not a Phase 2 blocker)
Wave 2 gate ran with a **temporary provider swap** — both agent YAMLs use `ROUTER_OPENAI_COMPATIBLE_API_KEY` (marked `# TEMP: gate test only`). The agents' real `step_plan` providers were NOT retested with async streaming. **Revert the swap and run one real-provider task completion.** This is provider-config, decoupled from run_engine deletion.

### Phase 2 — Delete dead plumbing
- Only after Phase 1 is proven in the UI (✅ done): remove `run_engine.py`, `chat_history_service.py`, run/stream/state routes.
- This is where the ~2,900 lines actually disappear.
- **Do it as a dependency-checked removal, NOT `rm`:** first grep for any remaining live imports of `run_engine` / `chat_history_service` / the routes; confirm thread-listing has moved to the SDK (`client.threads.search`) before deleting `state.py`; identify which `agents.py` endpoints are genuinely dead vs. still-needed companion routes. Delete, then run the full test suite + one real task run to confirm nothing broke.

### Phase 3 — Consolidate (optional)
- Move remaining custom domain endpoints into LangGraph Server custom routes; retire companion FastAPI if a single process is wanted. Or keep the hybrid — it's a fine end state.

## How the backend runs after migration

| Mode | Command | Infra |
|---|---|---|
| Local dev | `langgraph dev` | none (in-memory, port 2024, Studio UI) |
| Production | `langgraph up` | Postgres + Redis (Docker, auto-wired) |

We stop running `uvicorn solidcue.api.main:app` as *the* backend — either replace it (Phase 3) or demote it to a companion service (Phase 1).

## Open items to verify before/during Phase 0
- Exact Langfuse v4 OTel env/config (point C).
- Whether builders should expose compiled-without-checkpointer or the raw `StateGraph` for the server to compile.
- How current worker spawning maps to subgraphs vs. spawned child runs.
- Custom domain endpoints in `agents.py` that need to survive as companion routes or server custom routes.
