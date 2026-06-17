# Studio — Frontend Design

> Living design doc for the `studio/` web frontend. Status markers: ✅ Decided · 🟡 Proposed · 💡 Later · ❌ Out of scope.
> Decisions locked in §9. Nothing is built yet.

## 1. Purpose

A web console for the solidcue agent framework — the GUI counterpart to the Typer
CLI (`solidcue/app`). Both talk to the **same** `solidcue/api` HTTP layer, which
wraps `solidcue/services`. No business logic lives in the frontend.

## 2. Two hard constraints

### 2.1 Domain agnosticism (inherited from `CLAUDE.md`)
Core/UI must work for **any** agent, not just `resume_builder`.
- ❌ No hardcoded labels ("resume", "experience", "job description").
- ✅ Everything renders from config the API returns. Swapping `resume_builder` →
  `product_catalog` changes zero frontend code.

### 2.2 Additive-only backend ✅ (locked)
**No existing backend file is modified — ever.** All new endpoints live in
`solidcue/api/` and only *call* existing functions:
- `solidcue/services/*` — untouched (verified clean).
- `solidcue/core`, loaders (`agents/configs/loader`, `tools/loader`, `user/loader`),
  CLI — untouched.
- New v1 endpoints (delete agent, edit/toggle tool, profile, streaming) are
  feasible purely by *reusing* existing public functions (see §7). If a feature
  ever can't be done additively, it's cut — not forced via a backend edit.

## 3. Tech stack ✅

| Concern | Choice |
|---|---|
| Framework | React + TypeScript |
| Build/dev | Vite (API CORS already allows `localhost:5173`) |
| Routing | React Router |
| Server state | TanStack Query |
| **Styling** | **shadcn/ui + Tailwind** ✅ |
| Forms | React Hook Form + Zod |
| API types | Generated from FastAPI `/openapi.json` (single source of truth) |

## 3.4 Theme — Violet on Zinc, dark-first ✅

shadcn/ui CSS-variable theme. Primary = violet, neutrals = zinc. Ships **dark by
default** with a system-aware toggle in the top bar.

| Role | Light | Dark | Tailwind |
|---|---|---|---|
| `--background` | `#FFFFFF` | `#09090B` | white / zinc-950 |
| `--card` / surface | `#FAFAFA` | `#18181B` | zinc-50 / zinc-900 |
| `--foreground` | `#18181B` | `#FAFAFA` | zinc-900 / zinc-50 |
| `--muted-foreground` | `#71717A` | `#A1A1AA` | zinc-500 / zinc-400 |
| `--border` / `--input` | `#E4E4E7` | `#27272A` | zinc-200 / zinc-800 |
| `--primary` (accent) | `#7C3AED` | `#7C3AED` | violet-600 |
| `--primary` (hover/active) | `#6D28D9` | `#A78BFA` | violet-700 / violet-400 |
| `--primary-foreground` | `#FAFAFA` | `#FAFAFA` | zinc-50 |
| `--ring` (focus) | `#7C3AED` | `#7C3AED` | violet-600 |

- **Status colors** (run progress, approval, health dot): success `#10B981`
  (emerald-500), warning/approval `#F59E0B` (amber-500), error `#EF4444`
  (red-500), pending = `--muted-foreground`.
- Default `--radius`: `0.5rem`. Font: system UI stack (or Inter if added later 💡).

## 3.5 CLI parity — the baseline ✅

The frontend must do **everything the CLI does** (`solidcue/app`), with one
explicit exclusion. Audit of all 11 CLI commands:

| CLI command | Frontend home | API | Parity |
|---|---|---|---|
| `create-agent` | Agents → create wizard | `POST /agents` | ✅ |
| `list-agents` | Agents → list | `GET /agents` | ✅ |
| `run-agent` (+`--debug`) | Sessions → run console | `POST /agents/{k}/run`+`/resume` (+SSE) | ✅ |
| `snap` | — | `GET /state/*` | ❌ **excluded (CLI-only debug tool)** |
| `create-mcp-server` | MCP → add server | `POST /mcp/servers` | ✅ |
| `list-mcp-servers` | MCP → list | `GET /mcp/servers` | ✅ |
| `create-tool` (mcp/api/rag) | Tools → create | `POST /tools/{mcp,api,rag}` + discovery | ✅ |
| `list-tools` | Tools → list | `GET /tools` | ✅ |
| `setup-init` | Profile → edit | `PUT /profile` ← **new** | ✅ |
| `setup-view` | Profile → view | `GET /profile` ← **new** | ✅ |
| `setup-update` | Profile → edit | `PUT /profile` ← **new** | ✅ |

> Only gap to fill for full parity: the **3 profile endpoints** (additive, reuse
> `load/save_user_profile`). Everything else already has an endpoint.
> `run-agent --debug`'s token/metric table is **not** ported (part of `snap`-style
> debug, excluded). The studio shows clean run output only.

**Beyond CLI parity (extras, approved earlier):** delete-agent, edit/toggle-tool.
The CLI has no equivalent; these are frontend-only additions.

## 4. v1 scope — four pillars (in build priority order) ✅

> Priority from decision A: **1) Agents → 2) MCP → 3) Tools → 4) Sessions.**
> (Reading "build agent / build mcp / create agent / session" as: build the agent
> area, build MCP servers, create tools that agents use, then run sessions.
> Correct me if "create agent" meant something other than tool creation.)

```
studio/
  /agents                Pillar 1 — list, build/create wizard, detail, delete
  /mcp                   Pillar 2 — list servers, add (auto-discovers tools)
  /tools                 Pillar 3 — list, create MCP/API/RAG, edit/toggle
  /sessions/:thread_id   Pillar 4 — run console: streaming + approval loop
  /                      Dashboard (agents at a glance + quick run) 💡
  /profile               User profile (view/edit) 🟡
```

## 5. Features by pillar (mapped to endpoints)

### Pillar 1 — Agents
| Feature | Endpoint | Status |
|---|---|---|
| List agents (cards: model, tools, temps) | `GET /agents` | 🟡 |
| Create-agent wizard — 4 provider roles (brain/lite/reviewer/writer) + tool picker | `POST /agents` | 🟡 |
| Agent detail (read-only config) | `GET /agents` | 🟡 |
| **Delete agent** | `DELETE /agents/{key}` ← **new, additive** | 🟡 |
| Edit agent | _deferred_ | 💡 |

> Wizard mirrors the CLI's `prompt_provider_for_role` ×4. ⚠️ API keys POSTed in
> cleartext (backend writes to env file) — fine on localhost, needs auth+TLS before remote.

#### Agent card — Detailed ✅
The Agents list renders one **detailed** card per agent (matches `list-agents`).
```
┌────────────────────────────────┐
│ Resume Builder              ⋯   │   name (bold) + ⋯ menu
│ resume_builder · openai_compat  │   agent_key (mono) · provider.type
│ Build ATS-compatible resumes…   │   description (truncate ~2 lines)
│ brain    step-3.5-flash   0.3   │   provider.model / .temperature
│ lite     step-3.5-flash   0.1   │   lite_provider …
│ reviewer step-3.5-flash   0.1   │   reviewer_provider …
│ writer   step-3.5-flash   0.7   │   writer_provider …
│ 🔧 drive_upload  +9             │   tools[0] + (len-1) count
│             [ View ]  [ ▶ Run ] │   View → detail · Run → new Session
└────────────────────────────────┘
```
- **Role-model fallback:** `lite`/`reviewer`/`writer` reuse the brain `provider`'s
  model+temp when their config is null (same as the CLI). Show the resolved value;
  mark fallback rows subtly (e.g. muted "↳ inherits brain").
- **⋯ menu:** Delete (destructive, confirm dialog) → `DELETE /agents/{key}`.
  (Edit-agent is deferred — not in the menu yet.)
- **Run** (primary, violet) → opens a new Session for this agent.
- Empty tools → show "no tools" muted instead of the 🔧 row.

#### Create-agent wizard — 3 steps ✅
Full-page (`/agents/new`). Maps 1:1 to `CreateAgentInput` + the CLI's
`prompt_provider_for_role` ×4. Submit → `POST /agents` (409 = key exists → inline error).

**Step 1 · Identity**
- `name` (text) → `agent_key` derived live via `normalize_key`, shown read-only (mono).
- `description` (textarea, optional).

**Step 2 · Provider roles** — 4 role sub-forms: **Brain · Lite · Reviewer · Writer**.
Each role: `provider_type` (select: `openai_compatible` · `anthropic` · `openrouter`),
`base_url` (shown **only** when the type needs one — `openai_compatible`; others use
default), `api_key` (password), `model`, `temperature` (number).
- Prefills (match CLI): Lite/Reviewer/Writer default `model` = Brain's model;
  default temps **Lite 0.1 · Reviewer 0.1 · Writer 0.7** (Brain has no default).
- Writer is **optional** — an "inherit from Brain" toggle skips it (backend leaves
  `writer_provider` null when type/model/key absent).

**Step 3 · Tools + review**
- Tool multi-select from `GET /tools` (enabled only) → `selected_tools`.
- Review summary of all roles + tools → **Create**.

> ⚠️ API keys are typed here and POSTed in cleartext. Localhost-only (decision D).

### Pillar 2 — MCP servers
| Feature | Endpoint | Status |
|---|---|---|
| List servers | `GET /mcp/servers` | 🟡 |
| Add server (URL + auth) → auto-discovers tools, shows count | `POST /mcp/servers` | 🟡 |
| Refresh/preview discovery | `GET /tools/mcp-servers/{key}/discovered` | 🟡 |

#### MCP server card ✅ (maps to `MCPServerConfig`)
```
┌────────────────────────────────┐
│ Google Drive                    │  name (bold)
│ google_drive                    │  server_key (mono)
│ Exposes 12 tool(s): list, …     │  description (auto-built on create)
│ 🔗 https://mcp.example/sse      │  url (truncate, copy-on-click)
│ 🔒 bearer                       │  auth.type (lock icon; "none" → 🔓)
│            [ Discover tools ]   │  → opens discovery dialog
└────────────────────────────────┘
```
- **Discover tools** → `GET /tools/mcp-servers/{key}/discovered`, lists tools in a
  dialog; each row has a **"+ Create tool"** shortcut that opens the create-MCP-tool
  modal pre-filled with this server + tool (dedups already-registered ones, like the CLI).
- No ⋯ menu: there's no delete/edit endpoint for MCP servers (loader has none) — out of v1.

#### Add-MCP-server modal ✅ (maps to CLI `create-mcp-server`)
`name` → derived `server_key`; `url`; **auth builder**: `type`
(none/api_key/bearer/oauth) → if not none: `token_env` (validated env-var name,
*not* the raw key), `location` (header/query), then header_name+prefix **or**
param_name. Submit → `POST /mcp/servers` (auto-discovers, fills description).

### Pillar 3 — Tools
| Feature | Endpoint | Status |
|---|---|---|
| List tools | `GET /tools` | 🟡 |
| Create MCP tool (pick server → discover → pick → approval policy) | `GET /tools/mcp-servers`, `.../discovered`, `POST /tools/mcp` | 🟡 |
| Create API tool | `POST /tools/api` | 🟡 |
| Create RAG tool | `POST /tools/rag` | 🟡 |
| **Edit / toggle (enable-disable) tool** | `PUT /tools/{key}`, `PATCH /tools/{key}/enabled` ← **new, additive** | 🟡 |
| Delete tool | _deferred (loader has no delete; would need backend change)_ | ❌ |

#### Tool card ✅ (maps to `ToolConfig`)
```
┌────────────────────────────────┐
│ [MCP] Drive Upload      ⏻  ⋯   │  type badge + name + enable toggle + menu
│ drive_upload                    │  tool_key (mono)
│ Uploads a file to Drive…        │  description
│ server: google_drive · upload   │  type-specific line (see below)
│ approval: conditional · high 🔔 │  approval policy (🔔 if approval_prompt set)
└────────────────────────────────┘
```
- **Type badge** color-coded: `MCP` violet · `API` blue · `RAG` emerald.
- **Type-specific line:** `mcp` → `server: {mcp.server_key} · {mcp.tool_name}`;
  `api` → `{api.method} {api.base_url}`; `rag` → "RAG placeholder".
- **⏻ toggle** → `PATCH /tools/{key}/enabled` (load → flip `enabled` → save).
  Disabled tools render dimmed.
- **⋯ menu:** Edit → modal (`PUT /tools/{key}`). Delete deferred (no backend delete).

#### Create-tool modal ✅ (maps to CLI `create-tool`)
Type selector (`mcp`/`api`/`rag`) switches the form:
- **mcp:** pick server → discovered tools (deduped) → name/key derived → approval policy.
- **api:** name→key, description, base_url, method (GET/POST), auth builder, approval policy.
- **rag:** name→key, description, approval policy (placeholder — backend RAG is stubbed).
- **Approval policy** (shared): `approval_mode` (never/always/conditional),
  `approval_risk` (low/med/high), optional `approval_prompt`.

### Pillar 4 — Sessions (run console) — the centerpiece
Streaming run with the LangGraph **interrupt/approval** loop.

**Streaming ✅ (decision C):** new SSE endpoint streams live node progress.
- `POST /agents/{key}/run` and `/resume` already exist (blocking) — kept as fallback.
- **New, additive ✅ BUILT:** `POST /agents/{key}/stream` (SSE) — calls existing
  `build_agent_graph()` + `graph.stream(stream_mode="updates")`. POST (not a GET
  EventSource) so the run/resume body travels in the request; the frontend reads
  SSE frames from the response. Emits `start` → `node`* → terminal (`interrupt` |
  `completed`), or `error`. No existing file touched (`solidcue/api/streaming.py`).

**Approval loop UX:**
1. Start session → stream node-by-node progress (phase chips: classify → plan →
   decision → execution → …).
2. Stream ends in one of:
   - `completed` → render `output`.
   - `interrupted` → render **approval card** from the `interrupt` payload:
     `prompt`, `preview.{title,summary,sections[]}`; if `mode:"deterministic"` +
     `options[]` → option **buttons**, else free-text box.
3. User responds → resume (stream) with `thread_id` → back to step 1 until completed.

Faithful GUI of the CLI's `_prompt_interrupt_resume` loop, stateless over `thread_id`.

#### Approval card ✅ (renders the `interrupt` payload)
Appears inline in the chat thread when a stream ends with `status:"interrupted"`.
Payload shape (from the CLI loop): `{ mode, prompt, preview{title,summary,sections[{label,content}]}, options[] }`.
```
┌──────────────────────────────────────┐
│ ⚠ Approval required                   │  amber header
│ <prompt>                              │
│ ┌──────────────────────────────────┐ │
│ │ <preview.title>                  │ │  bold
│ │ <preview.summary>                │ │
│ │ Tool input:                      │ │  section.label
│ │   { "id": "…", … }               │ │  section.content — JSON pretty-printed
│ └──────────────────────────────────┘ │     if parseable (mirrors CLI)
│ deterministic → [APPROVE] [REJECT] …  │  buttons from options[] (uppercased)
│ free-text     → [ textarea ] [ Send ] │  otherwise
└──────────────────────────────────────┘
```
- `mode == "deterministic"` + `options[]` → render one **button per option**
  (uppercased); APPROVE styled primary/violet, REJECT destructive/red. The chosen
  option string is the `resume_value`.
- Otherwise → free-text box; its content is the `resume_value`.
- Submit → resume the stream with `{thread_id, resume_value}` → loop continues.
- `preview.sections` render in order; a section labeled "Tool input" pretty-prints
  JSON (falls back to plain text) — same special-case as the CLI.

### Profile (cross-pillar)
| Feature | Endpoint | Status |
|---|---|---|
| View/edit profile (location, timezone, prefs) | `GET/PUT /profile` ← **new, additive** (reuses `load/save_user_profile`) | 🟡 |

### State inspector (`snap`) — ❌ EXCLUDED from frontend
The CLI `snap` debug command is **not** ported to the studio. It's a
developer-only state/debug tool; the CLI and `/docs` cover it. The
`GET /state/*` endpoints stay in the API but the studio ships no UI for them.
| Feature | Endpoint | Status |
|---|---|---|
| Live state by thread / per-node metrics | `GET /state/*` | ❌ CLI-only |

## 6. Cross-cutting
- **API client:** typed, generated from `/openapi.json`. 🟡
- **Config:** `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`). 🟡
- **Error handling:** surface API `detail` (400/404/409/502). 🟡
- **Auth:** ❌ out of scope for v1 local (decision D). Needed before remote.
- **Dark mode / theming:** 💡 (shadcn makes this cheap).

## 7. New backend endpoints for v1 — all additive ✅
Each verified feasible by reusing existing public functions; **no existing file is edited.**

| New endpoint | Reuses (existing, unmodified) | New file |
|---|---|---|
| `DELETE /agents/{key}` | `get_agent_path`, `get_persona_path`, `get_skill_path`, `get_tools_path` → unlink | `api/routes/agents.py` (+ maybe `api/admin/agent_files.py`) |
| `PUT /tools/{key}`, `PATCH /tools/{key}/enabled` | `load_tool` + `save_tool` | `api/routes/tools.py` |
| `GET/PUT /profile` | `load_user_profile` + `save_user_profile` | `api/routes/profile.py` |
| `POST /agents/{key}/stream` (SSE) | `build_agent_graph`, `graph.stream` | `api/routes/agents.py` + `api/streaming.py` |

> ❌ Still genuinely blocked (need a real backend change, so deferred): **delete
> tool** (no delete in `tools/loader`), **edit agent** (regen of env keys/markdown
> is non-trivial). Out of v1.

## 8. Build order ✅
0. ✅ **DONE** — v1 backend endpoints (§7) added in `api/`, verified, `services/` clean.
1. ✅ **DONE** — `studio/` scaffolded (Vite + React + TS + shadcn/Tailwind v4),
   violet/zinc dark-first theme, sidebar shell, typed API client, TanStack Query,
   `/api`→:8000 proxy, health dot. Verified in browser.
2. **Pillar 1 — Agents:** ✅ DONE — list + detailed card + delete dialog +
   3-step create wizard (`/agents/new`), all verified in browser.
3. **Pillar 2 — MCP:** list → add server (with discovery).
4. **Pillar 3 — Tools:** list → create (MCP/API/RAG) → edit/toggle.
5. **Pillar 4 — Sessions:** SSE streaming run + approval loop. (No state/debug panel — excluded.)
6. Profile (CLI parity), dashboard, dark mode polish.

## 8.5 Layout — LOCKED ✅

### Global shell — left sidebar
Persistent, collapsible left nav. Top bar shows app name, API health dot, theme toggle.
```
┌──────────────────────────────────────────────┐
│ ☰ solidcue studio         ● API ok    ◐ theme │
├──────────┬───────────────────────────────────┤
│ ▸Agents  │   <page content>        [ + New ]  │
│  MCP     │                                    │
│  Tools   │                                    │
│  Sessions│                                    │
│  Profile │                                    │
└──────────┴───────────────────────────────────┘
```
- Nav items: **Agents · MCP · Tools · Sessions · Profile** (Dashboard 💡 later).
- List pages = header + `+ New` button + cards/table.

### Create/edit flows — modal + wizard
- **Modal dialog** for the simple forms: create MCP server, create tool
  (mcp/api/rag), edit/toggle tool, profile edit.
- **Full-page wizard** (`/agents/new`) for create-agent — it has 4 provider roles
  (brain/lite/reviewer/writer) + tool picker, too much for a modal. Steps:
  1. Identity (name → derived key, description)
  2. Provider roles ×4 (type, base_url, api_key, model, temperature)
  3. Tool picker + review → submit.

### Sessions run console — two-pane
```
 Session · <agent>
┌───────────────────┬────────────────────┐
│ chat thread       │ Run progress (SSE)  │
│  you: ...         │  ✓ classifier      │
│  agent: ...       │  ✓ planning        │
│  ⚠ approval card  │  ⟳ execution       │
│   [Approve][Reject]│  · synthesis       │
│ [ type... ][Send] │  · validation      │
└───────────────────┴────────────────────┘
```
- **Left pane:** chat thread (user/agent turns) + inline approval cards
  (from the `interrupt` payload: prompt, preview, option buttons or free text).
- **Right pane ("Run progress"):** live node rail driven by SSE `updates`. Node
  order mirrors the backend execution order: `initialize → classifier → discovery
  → planning → decision → execution → reflection → synthesis → validation →
  validation_hhem → final_output`. Status icons only (done/active/pending) — **no
  token/metric numbers** (that's the excluded `snap`/debug view).

## 9. Decisions — LOCKED
- **A. Focus:** Both, builder-first. Order: Agents → MCP → Tools → Sessions. ✅
- **B. Styling:** shadcn/ui + Tailwind. ✅
- **C. Run mode:** Streaming now (additive SSE endpoint). ✅
- **D. Auth:** Out of scope for v1 (local only). ✅
- **E. Backend gaps in v1:** delete agent, edit/toggle tool, user-profile route —
  all additive (§7). Delete-tool & edit-agent deferred. ✅
- **F. Backend rule:** Additive-only — no existing file is ever modified. ✅
- **G. Layout:** Left sidebar shell · modal forms + full-page agent wizard ·
  two-pane sessions (chat + run-progress rail). ✅ (see §8.5)
- **H. Theme:** Violet on Zinc, dark-first + system toggle. ✅ (see §3.4)
```
