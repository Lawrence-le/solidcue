import { useCallback, useEffect, useRef, useState } from "react"
import { useNavigate, useSearchParams } from "react-router-dom"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  AlertCircle,
  Bot,
  CheckCircle2,
  CircleDot,
  Loader2,
  Send,
  Square,
} from "lucide-react"
import { api, ApiError, streamAgent } from "@/lib/api"
import type { InterruptPayload, StreamEvent } from "@/lib/types"
import { MarkdownContent } from "@/components/MarkdownContent"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type RunState = "idle" | "streaming" | "interrupted" | "completed" | "error" | "disconnected" | "cancelled"

type ChatMessageInput =
  | { role: "user"; content: string }
  | { role: "assistant"; content: string }
  | { role: "interrupt"; payload: InterruptPayload; threadId: string }
  | { role: "error"; content: string }
  | { role: "system"; content: string }

type ChatMessage = ChatMessageInput & { id: string }
type PersistedChatHistoryEntry = { role?: string; content?: string }

interface NodeTokens {
  input: number
  output: number
  total: number
}

interface NodeEvent {
  node: string
  phase: string | null
  ts: number
  status: "running" | "done"
  tokens?: NodeTokens
}

let _msgId = 0
function msgId() {
  return String(++_msgId)
}

function mapRemoteRunStatus(status: string | undefined): RunState {
  if (status === "running") return "streaming"
  if (status === "interrupted") return "interrupted"
  if (status === "completed") return "completed"
  if (status === "error") return "error"
  if (status === "disconnected") return "disconnected"
  if (status === "cancelled") return "cancelled"
  return "idle"
}

// ---------------------------------------------------------------------------
// Approval card
// ---------------------------------------------------------------------------

function ApprovalCard({
  payload,
  onSubmit,
  disabled,
}: {
  payload: InterruptPayload
  onSubmit: (value: string) => void
  disabled: boolean
}) {
  const [custom, setCustom] = useState("")
  const hasOptions = (payload.options ?? []).length > 0

  return (
    <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4 space-y-3">
      <div className="flex items-start gap-2">
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
        <div className="min-w-0 flex-1 space-y-1">
          <p className="text-sm font-medium text-amber-300">Approval required</p>
          {payload.prompt && (
            <p className="text-sm text-muted-foreground">{payload.prompt}</p>
          )}
        </div>
      </div>

      {payload.preview && (
        <div className="rounded border border-border bg-card p-3 space-y-1.5">
          {payload.preview.title && <p className="text-xs font-medium">{payload.preview.title}</p>}
          {payload.preview.summary && (
            <p className="text-xs text-muted-foreground">{payload.preview.summary}</p>
          )}
          {(payload.preview.sections ?? []).map((sec, i) => (
            <div key={i} className="text-xs">
              {sec.label && <span className="font-medium">{sec.label}: </span>}
              <span className="text-muted-foreground">{sec.content}</span>
            </div>
          ))}
        </div>
      )}

      {hasOptions && (
        <div className="flex flex-wrap gap-2">
          {payload.options!.map((opt) => (
            <Button
              key={opt}
              size="sm"
              variant="outline"
              disabled={disabled}
              onClick={() => onSubmit(opt)}
              className="border-amber-500/30 text-xs hover:bg-amber-500/10"
            >
              {opt}
            </Button>
          ))}
        </div>
      )}

      <div className="flex gap-2">
        <Textarea
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          placeholder={hasOptions ? "Or type a custom response…" : "Type your response…"}
          rows={2}
          className="text-sm"
          disabled={disabled}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && custom.trim()) {
              e.preventDefault()
              onSubmit(custom.trim())
              setCustom("")
            }
          }}
        />
        <Button
          size="icon"
          disabled={disabled || !custom.trim()}
          onClick={() => { onSubmit(custom.trim()); setCustom("") }}
          className="shrink-0 self-end"
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Node progress rail
// ---------------------------------------------------------------------------

function NodeRail({ events }: { events: NodeEvent[] }) {
  if (events.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center px-4">
        <p className="text-xs text-muted-foreground">Node events appear here during a run.</p>
      </div>
    )
  }
  return (
    <div className="space-y-1 p-3">
      {events.map((ev, i) => (
        <div key={i} className="flex items-start gap-2 py-1">
          {ev.status === "running" ? (
            <Loader2 className="mt-0.5 h-3.5 w-3.5 shrink-0 animate-spin text-primary" />
          ) : (
            <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-green-500" />
          )}
          <div className="min-w-0 flex-1">
            <div className="font-mono text-xs font-medium">{ev.node}</div>
            {ev.phase && (
              <div className="text-xs text-muted-foreground">{ev.phase}</div>
            )}
            <div className="flex items-center gap-2">
              <div className="text-xs text-muted-foreground/50 tabular-nums">
                {new Date(ev.ts).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                })}
              </div>
              {ev.tokens && (
                <div className="text-xs text-muted-foreground/60 tabular-nums">
                  {ev.tokens.input}↑ {ev.tokens.output}↓
                </div>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// SessionsPage
// ---------------------------------------------------------------------------

export function SessionsPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const qc = useQueryClient()

  const [agentKey, setAgentKey] = useState<string>("")
  const [threadId, setThreadId] = useState<string | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [nodeEvents, setNodeEvents] = useState<NodeEvent[]>([])
  const [runState, setRunState] = useState<RunState>("idle")
  const [resumable, setResumable] = useState(false)
  const [userInput, setUserInput] = useState("")
  const [loadingSession, setLoadingSession] = useState(false)

  const abortRef = useRef<AbortController | null>(null)
  const stopIntentionalRef = useRef(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const nodeRailEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const streamingAssistantIdRef = useRef<string | null>(null)

  useEffect(() => {
    if (!userInput && inputRef.current) {
      inputRef.current.style.height = ""
    }
  }, [userInput])

  const { data: agents } = useQuery({ queryKey: ["agents"], queryFn: api.listAgents })

  const loadThreadSnapshot = useCallback(async (targetThreadId: string) => {
    const [stateRes, runRes] = await Promise.all([
      api.liveState(targetThreadId, ["chat_history", "user_input", "final_response", "agent_key"]),
      api.runStatus(targetThreadId),
    ])

    const state = stateRes.state
    const msgs: ChatMessage[] = []
    const chatHistory = Array.isArray(state.chat_history) ? state.chat_history as PersistedChatHistoryEntry[] : []
    for (const entry of chatHistory) {
      if (entry?.role === "user" && typeof entry.content === "string") {
        msgs.push({ role: "user", content: entry.content, id: msgId() })
      } else if (entry?.role === "assistant" && typeof entry.content === "string") {
        msgs.push({ role: "assistant", content: entry.content, id: msgId() })
      }
    }
    if (msgs.length === 0 && state.user_input) msgs.push({ role: "user", content: String(state.user_input), id: msgId() })
    if (msgs.length === 1 && state.final_response) msgs.push({ role: "assistant", content: String(state.final_response), id: msgId() })
    if (runRes.status === "interrupted") {
      try {
        const interruptRes = await api.getInterrupt(targetThreadId)
        if (interruptRes.interrupt) {
          msgs.push({ role: "interrupt", payload: interruptRes.interrupt, threadId: targetThreadId, id: msgId() })
        }
      } catch {
        // Best effort — interrupt payload unavailable
      }
    } else if (runRes.status === "running") {
      msgs.push({ role: "system", content: "Run still in progress. Live updates were interrupted by refresh.", id: msgId() })
    } else if (runRes.status === "disconnected") {
      msgs.push({ role: "system", content: "Previous run was interrupted when the browser disconnected.", id: msgId() })
      try {
        const r = await api.isResumable(targetThreadId)
        setResumable(r.resumable)
      } catch { /* best effort */ }
    } else if (runRes.status === "cancelled") {
      try {
        const r = await api.isResumable(targetThreadId)
        setResumable(r.resumable)
      } catch { /* best effort */ }
    } else if (runRes.status === "error" && runRes.error) {
      msgs.push({ role: "error", content: runRes.error, id: msgId() })
    }
    if (msgs.length === 0) msgs.push({ role: "system", content: `Session ${targetThreadId.slice(0, 8)} loaded.`, id: msgId() })
    if (state.agent_key && typeof state.agent_key === "string") setAgentKey(state.agent_key)
    setMessages(msgs)
    setRunState(mapRemoteRunStatus(runRes.status))
  }, [])

  // Load session when ?thread= param changes
  const threadParam = searchParams.get("thread")
  useEffect(() => {
    if (!threadParam) {
      // New session — reset everything
      abortRef.current?.abort()
      streamingAssistantIdRef.current = null
      setThreadId(null)
      setRunId(null)
      setMessages([])
      setNodeEvents([])
      setRunState("idle")
      setResumable(false)
      setUserInput("")
      return
    }
    if (threadParam === threadId) return

    abortRef.current?.abort()
    streamingAssistantIdRef.current = null
    setThreadId(threadParam)
    setRunId(null)
    setMessages([])
    setNodeEvents([])
    setRunState("idle")
    setResumable(false)
    setLoadingSession(true)

    loadThreadSnapshot(threadParam)
      .catch(() => {
        setMessages([{ role: "system", content: `Session ${threadParam.slice(0, 8)} loaded.`, id: msgId() }])
      })
      .finally(() => setLoadingSession(false))
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threadParam, loadThreadSnapshot])

  useEffect(() => {
    if (!threadId || runState !== "streaming" || abortRef.current) return

    const interval = window.setInterval(async () => {
      try {
        const status = await api.runStatus(threadId)
        const mapped = mapRemoteRunStatus(status.status)
        if (mapped !== "streaming") {
          await loadThreadSnapshot(threadId)
        }
      } catch {
        // Best effort polling only.
      }
    }, 2000)

    return () => window.clearInterval(interval)
  }, [threadId, runState, loadThreadSnapshot])

  useEffect(() => {
    nodeRailEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [nodeEvents])

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [])
  useEffect(() => { scrollToBottom() }, [messages, scrollToBottom])

  function addMessage(msg: ChatMessageInput) {
    setMessages((prev) => [...prev, { ...msg, id: msgId() } as ChatMessage])
  }

  function ensureStreamingAssistantMessage() {
    if (streamingAssistantIdRef.current) return streamingAssistantIdRef.current
    const id = msgId()
    streamingAssistantIdRef.current = id
    setMessages((prev) => [...prev, { role: "assistant", content: "", id }])
    return id
  }

  function appendStreamingAssistantDelta(delta: string) {
    const id = ensureStreamingAssistantMessage()
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === id && msg.role === "assistant"
          ? { ...msg, content: msg.content + delta }
          : msg,
      ),
    )
  }

  function finalizeStreamingAssistant(output: string) {
    const id = streamingAssistantIdRef.current
    if (!id) {
      if (output) addMessage({ role: "assistant", content: output })
      return
    }
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === id && msg.role === "assistant"
          ? { ...msg, content: msg.content || output }
          : msg,
      ),
    )
    streamingAssistantIdRef.current = null
  }

  function markNodeDone(node: string) {
    setNodeEvents((prev) =>
      prev.map((ev) => (ev.node === node && ev.status === "running" ? { ...ev, status: "done" } : ev)),
    )
  }

  const handleEvent = useCallback(
    (e: StreamEvent) => {
      if (e.event === "start") {
        const tid = e.data.thread_id
        setThreadId(tid)
        setRunId(e.data.run_id)
        navigate(`/sessions?thread=${tid}`, { replace: true })
      } else if (e.event === "message_start") {
        ensureStreamingAssistantMessage()
      } else if (e.event === "message_delta") {
        appendStreamingAssistantDelta(e.data.delta)
      } else if (e.event === "node") {
        const tokens = e.data.tokens ?? undefined
        setNodeEvents((prev) => {
          const last = prev[prev.length - 1]
          if (last?.status === "running") {
            return [
              ...prev.slice(0, -1),
              { ...last, status: "done" as const },
              { node: e.data.node, phase: e.data.phase, ts: Date.now(), status: "running" as const, tokens },
            ]
          }
          return [...prev, { node: e.data.node, phase: e.data.phase, ts: Date.now(), status: "running" as const, tokens }]
        })
      } else if (e.event === "interrupt") {
        markNodeDone(nodeEvents[nodeEvents.length - 1]?.node ?? "")
        setRunState("interrupted")
        setThreadId(e.data.thread_id)
        abortRef.current = null
        addMessage({ role: "interrupt", payload: e.data.interrupt, threadId: e.data.thread_id })
      } else if (e.event === "completed") {
        setNodeEvents((prev) => prev.map((ev) => ({ ...ev, status: "done" })))
        setRunState("completed")
        abortRef.current = null
        finalizeStreamingAssistant(e.data.output)
        qc.invalidateQueries({ queryKey: ["threads"] })
      } else if (e.event === "cancelled") {
        setNodeEvents((prev) => prev.map((ev) => ({ ...ev, status: "done" })))
        setRunState("cancelled")
        setResumable(true)
        streamingAssistantIdRef.current = null
        abortRef.current = null
      } else if (e.event === "error") {
        setRunState("error")
        streamingAssistantIdRef.current = null
        abortRef.current = null
        addMessage({ role: "error", content: e.data.message })
      }
    },
    [nodeEvents, qc, navigate],
  )

  async function handleContinue() {
    if (!agentKey || !threadId) return
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setRunState("streaming")
    setNodeEvents([])
    try {
      await streamAgent(agentKey, { thread_id: threadId }, handleEvent, ctrl.signal)
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        streamingAssistantIdRef.current = null
        abortRef.current = null
        addMessage({ role: "error", content: (err as ApiError).message ?? String(err) })
        setRunState("error")
      } else {
        streamingAssistantIdRef.current = null
        abortRef.current = null
        if (stopIntentionalRef.current) {
          stopIntentionalRef.current = false
        } else {
          setRunState("idle")
        }
      }
    }
  }

  async function startRun(input: string, resumeValue?: string) {
    if (!agentKey) return
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setRunState("streaming")
    setNodeEvents([])

    const body = resumeValue
      ? { thread_id: threadId ?? undefined, resume_value: resumeValue }
      : { thread_id: threadId ?? undefined, user_input: input }

    try {
      await streamAgent(agentKey, body, handleEvent, ctrl.signal)
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        streamingAssistantIdRef.current = null
        abortRef.current = null
        addMessage({ role: "error", content: (err as ApiError).message ?? String(err) })
        setRunState("error")
      } else {
        streamingAssistantIdRef.current = null
        abortRef.current = null
        if (stopIntentionalRef.current) {
          stopIntentionalRef.current = false
        } else {
          setRunState("idle")
        }
      }
    }
  }

  function handleSend() {
    if (!userInput.trim() || !agentKey || runState === "streaming") return
    const text = userInput.trim()
    setUserInput("")
    addMessage({ role: "user", content: text })
    startRun(text)
  }

  function handleResume(value: string) {
    setRunState("streaming")
    startRun("", value)
  }

  async function handleStop() {
    stopIntentionalRef.current = true
    abortRef.current?.abort()
    abortRef.current = null
    streamingAssistantIdRef.current = null
    setRunState("cancelled")
    setResumable(true)
    setNodeEvents((prev) => prev.map((ev) => ({ ...ev, status: "done" })))
    if (agentKey && runId) {
      try { await api.cancelRun(agentKey, runId) } catch { /* best effort */ }
    }
    if (threadId) {
      try { await loadThreadSnapshot(threadId) } catch { /* best effort */ }
    }
  }

  async function handleResumeFromCancel() {
    if (!agentKey || !threadId) return
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setRunState("streaming")
    setResumable(false)
    setNodeEvents([])
    try {
      await streamAgent(agentKey, { thread_id: threadId }, handleEvent, ctrl.signal)
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        streamingAssistantIdRef.current = null
        abortRef.current = null
        addMessage({ role: "error", content: (err as ApiError).message ?? String(err) })
        setRunState("error")
      } else {
        streamingAssistantIdRef.current = null
        abortRef.current = null
        setRunState("idle")
      }
    }
  }

  const streaming = runState === "streaming"
  const canSend = !!agentKey && !!userInput.trim() && !streaming && runState !== "interrupted" && runState !== "disconnected" && runState !== "cancelled"

  return (
    <div className="-m-6 flex h-[calc(100vh-3.5rem)] flex-col overflow-hidden">
      {/* Chat pane + node rail */}
      <div className="flex flex-1 min-h-0">
        {/* Chat pane */}
        <div className="flex flex-col flex-1 min-w-0 border-r">
          {/* Agent selector bar */}
          <div className="flex items-center gap-3 px-4 py-2.5 border-b bg-muted/30 shrink-0">
            <Bot className="h-4 w-4 shrink-0 text-muted-foreground" />
            {threadId ? (
              <>
                <span className="text-sm font-medium">
                  {agents?.find((a) => a.agent_key === agentKey)?.name ?? agentKey}
                </span>
                <Badge variant="outline" className="ml-auto font-mono text-xs">
                  {threadId.slice(0, 8)}
                </Badge>
              </>
            ) : (
              <Select value={agentKey} onValueChange={setAgentKey} disabled={streaming}>
                <SelectTrigger className="h-8 w-52 text-sm">
                  <SelectValue placeholder="Select agent…" />
                </SelectTrigger>
                <SelectContent>
                  {(agents ?? []).map((a) => (
                    <SelectItem key={a.agent_key} value={a.agent_key}>
                      {a.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>

          {/* Messages */}
          <div className="flex-1 min-h-0 overflow-y-auto px-4 sm:px-8 lg:px-16 [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border [&::-webkit-scrollbar-track]:bg-transparent">
            <div className="space-y-4 py-4">
              {messages.length === 0 && !streaming && !loadingSession && (
                <div className="flex flex-col items-center justify-center py-20 text-center">
                  <CircleDot className="mb-3 h-8 w-8 text-muted-foreground/30" />
                  <p className="text-sm font-medium">No messages yet</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {agentKey ? "Type a message below to start." : "Select an agent to begin."}
                  </p>
                </div>
              )}

              {messages.map((msg) => (
                <div key={msg.id}>
                  {msg.role === "user" && (
                    <div className="flex justify-end">
                      <div className="max-w-[80%] break-words rounded-2xl rounded-br-sm bg-muted px-4 py-2.5 text-sm text-foreground">
                        {msg.content}
                      </div>
                    </div>
                  )}
                  {msg.role === "assistant" && (
                    <div className="flex min-w-0">
                      <div className="min-w-0 max-w-[80%] overflow-hidden px-4 py-2.5">
                        <MarkdownContent content={msg.content} />
                      </div>
                    </div>
                  )}
                  {msg.role === "interrupt" && (
                    <ApprovalCard
                      payload={msg.payload}
                      onSubmit={handleResume}
                      disabled={runState !== "interrupted"}
                    />
                  )}
                  {msg.role === "error" && (
                    <div className="flex items-center gap-2 px-1 text-sm text-destructive">
                      <AlertCircle className="h-4 w-4 shrink-0" />
                      {msg.content}
                    </div>
                  )}
                  {msg.role === "system" && (
                    <div className="flex flex-col items-center gap-2">
                      <p className="text-center text-xs text-muted-foreground">{msg.content}</p>
                      {runState === "disconnected" && resumable && (
                        <Button size="sm" variant="outline" onClick={handleContinue}>
                          Continue from checkpoint
                        </Button>
                      )}
                    </div>
                  )}
                </div>
              ))}

              {loadingSession && (
                <div className="flex items-center justify-center gap-2 py-6 text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="text-sm">Loading session from DB…</span>
                </div>
              )}

              {streaming && (
                <div className="flex">
                  <div className="flex items-center gap-1.5 rounded-2xl rounded-bl-sm bg-muted px-4 py-2.5">
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">Thinking…</span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Cancelled banner */}
          {runState === "cancelled" && resumable && (
            <div className="shrink-0 flex items-center justify-between gap-3 px-4 sm:px-8 lg:px-16 py-2 border-t bg-muted/30">
              <p className="text-xs text-muted-foreground">Run stopped. You can resume from where it left off.</p>
              <Button size="sm" variant="outline" onClick={handleResumeFromCancel}>
                Resume
              </Button>
            </div>
          )}

          {/* Input */}
          <div className="shrink-0 px-4 sm:px-8 lg:px-16 pb-4 pt-2">
            <div
              className="relative rounded-[24px] border border-border bg-card pl-4 pr-2 py-0.5 transition-all focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/30"
            >
              <div className="flex items-center gap-2">
                <textarea
                  ref={inputRef}
                  value={userInput}
                  onChange={(e) => {
                    setUserInput(e.target.value)
                    const el = e.target
                    el.style.height = "auto"
                    el.style.height = `${Math.min(el.scrollHeight, 150)}px`
                  }}
                  placeholder={agentKey ? "Type your message…" : "Select an agent first"}
                  rows={1}
                  disabled={!agentKey || streaming || runState === "interrupted" || runState === "disconnected" || runState === "cancelled"}
                  className="flex-1 resize-none bg-transparent py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none disabled:opacity-50 leading-relaxed max-h-[150px] overflow-y-auto"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      if (canSend) handleSend()
                    }
                  }}
                />
                <div className="flex items-center my-0.5">
                  {streaming ? (
                    <button
                      type="button"
                      onClick={handleStop}
                      className="flex h-7 w-7 items-center justify-center rounded-full bg-muted hover:text-destructive transition-colors"
                    >
                      <Square className="h-3 w-3 fill-current" />
                    </button>
                  ) : (
                    <button
                      type="button"
                      disabled={!canSend}
                      onClick={handleSend}
                      className="flex h-7 w-7 items-center justify-center rounded-full bg-primary text-primary-foreground transition-all hover:opacity-90 disabled:bg-muted disabled:text-muted-foreground active:scale-95"
                    >
                      <Send className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              </div>
            </div>
            <p className="mt-1.5 px-1 text-xs text-muted-foreground/50">Shift+Enter for new line</p>
          </div>
        </div>

        {/* Node progress rail */}
        <div className="flex w-48 shrink-0 flex-col">
          <div className="flex shrink-0 items-center gap-2 border-b bg-muted/30 px-3 py-2.5">
            <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Trace
            </span>
            {streaming && (
              <Badge variant="secondary" className="ml-auto text-xs">
                <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                Running
              </Badge>
            )}
            {runState === "completed" && (
              <Badge variant="outline" className="ml-auto border-green-500/30 text-xs text-green-500">
                Done
              </Badge>
            )}
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border [&::-webkit-scrollbar-track]:bg-transparent">
            <NodeRail events={nodeEvents} />
            <div ref={nodeRailEndRef} />
          </div>
        </div>
      </div>
    </div>
  )
}
