import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle,
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  Circle,
  Loader2,
  Send,
  Settings2,
  Square,
} from "lucide-react";
import { api, ApiError } from "@/lib/api"
import {
  lgCancelRun,
  lgCreateThread,
  loadThreadMapping,
  persistThreadMapping,
  streamLangGraph,
} from "@/lib/lgClient";
import type {
  InterruptPayload,
  StreamEvent,
  UpdateProfileRequest,
  UserProfileConfig,
} from "@/lib/types";
import {
  PROVIDER_META,
  type ProviderType,
} from "@/lib/agent-config";
import { MarkdownContent } from "@/components/MarkdownContent";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type RunState =
  | "idle"
  | "streaming"
  | "interrupted"
  | "completed"
  | "error"
  | "cancelled";

type ChatMessageInput =
  | { role: "user"; content: string }
  | { role: "assistant"; content: string }
  | { role: "interrupt"; payload: InterruptPayload; threadId: string }
  | { role: "error"; content: string }
  | { role: "system"; content: string };

type ChatMessage = ChatMessageInput & { id: string };
type PersistedChatHistoryEntry = { role?: string; content?: string };

interface RouterSettingsForm {
  provider_type: ProviderType;
  base_url: string;
  api_key: string;
  model: string;
  temperature: string;
}

interface NodeTokens {
  input: number;
  output: number;
  total: number;
}

interface NodeEvent {
  node: string;
  phase: string | null;
  ts: number;
  status: "running" | "done";
  tokens?: NodeTokens;
}

interface SubagentStep {
  agentKey: string;
  subTask: string;
  stepIndex: number;
  stepCount: number;
  status: "pending" | "running" | "completed" | "failed" | "interrupted";
  output: string;
}

let _msgId = 0;
function msgId() {
  return String(++_msgId);
}

function mapRemoteRunStatus(status: string | undefined): RunState {
  if (status === "running") return "streaming";
  if (status === "interrupted") return "interrupted";
  if (status === "completed") return "completed";
  if (status === "error") return "error";
  if (status === "cancelled") return "cancelled";
  return "idle";
}

// ---------------------------------------------------------------------------
// Approval card
// ---------------------------------------------------------------------------

function ApprovalCard({
  payload,
  onSubmit,
  disabled,
}: {
  payload: InterruptPayload;
  onSubmit: (value: string) => void;
  disabled: boolean;
}) {
  const [custom, setCustom] = useState("");
  const hasOptions = (payload.options ?? []).length > 0;

  return (
    <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4 space-y-3">
      <div className="flex items-start gap-2">
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
        <div className="min-w-0 flex-1 space-y-1">
          <p className="text-sm font-medium text-amber-300">
            Approval required
          </p>
          {payload.prompt && (
            <p className="text-sm text-muted-foreground">{payload.prompt}</p>
          )}
        </div>
      </div>

      {payload.preview && (
        <div className="rounded border border-border bg-card p-3 space-y-1.5">
          {payload.preview.title && (
            <p className="text-xs font-medium">{payload.preview.title}</p>
          )}
          {payload.preview.summary && (
            <p className="text-xs text-muted-foreground">
              {payload.preview.summary}
            </p>
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
          placeholder={
            hasOptions ? "Or type a custom response…" : "Type your response…"
          }
          rows={2}
          className="text-sm"
          disabled={disabled}
          onKeyDown={(e) => {
            if (
              e.key === "Enter" &&
              (e.metaKey || e.ctrlKey) &&
              custom.trim()
            ) {
              e.preventDefault();
              onSubmit(custom.trim());
              setCustom("");
            }
          }}
        />
        <Button
          size="icon"
          disabled={disabled || !custom.trim()}
          onClick={() => {
            onSubmit(custom.trim());
            setCustom("");
          }}
          className="shrink-0 self-end"
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Node progress rail
// ---------------------------------------------------------------------------

function StepHistory({ events }: { events: NodeEvent[] }) {
  if (events.length === 0) return null;

  return (
    <div className="space-y-1.5 px-1">
      <p className="text-[10px] font-medium uppercase tracking-[0.18em] text-muted-foreground/70">
        Steps
      </p>
      <div className="space-y-1">
        {events.map((event, index) => {
          const isRunning = event.status === "running";
          const timestamp = new Date(event.ts).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          });

          return (
            <div
              key={`${event.node}-${event.ts}-${index}`}
              className={cn(
                "flex items-start gap-2 px-2 py-1",
                isRunning
                  ? "text-foreground/75"
                  : "text-muted-foreground",
              )}
            >
              <div className="mt-0.5">
                {isRunning ? (
                  <Loader2 className="h-3 w-3 animate-spin text-primary" />
                ) : (
                  <CheckCircle2 className="h-3 w-3 text-muted-foreground/70" />
                )}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[11px] leading-none">
                    {event.node}
                  </span>
                  <span className="text-[10px] leading-none text-muted-foreground/60">
                    {timestamp}
                  </span>
                </div>
                {event.phase && (
                  <p className="mt-0.5 text-[10px] leading-tight text-muted-foreground/75">
                    {event.phase}
                  </p>
                )}
                {event.tokens && (
                  <div className="mt-0.5 text-[10px] tabular-nums text-muted-foreground/60">
                    {event.tokens.input}↑ {event.tokens.output}↓
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SubagentActivity({
  steps,
  intro,
  resolveName,
}: {
  steps: SubagentStep[];
  intro: string;
  resolveName: (agentKey: string) => string;
}) {
  if (steps.length === 0 && !intro) return null;

  return (
    <div className="space-y-1.5 px-1">
      <p className="text-[10px] font-medium uppercase tracking-[0.18em] text-muted-foreground/70">
        Plan
      </p>
      {intro && (
        <p className="text-xs leading-snug text-foreground/80">{intro}</p>
      )}
      <div className="space-y-1.5">
        {[...steps]
          .sort((a, b) => a.stepIndex - b.stepIndex)
          .map((step) => {
            const isRunning = step.status === "running";
            const isPending = step.status === "pending";
            const failed = step.status === "failed" || step.status === "interrupted";
            return (
              <div
                key={step.stepIndex}
                className={cn(
                  "rounded-md border border-border/60 px-2.5 py-1.5",
                  isPending ? "bg-muted/10 opacity-70" : "bg-muted/20",
                )}
              >
                <div className="flex items-center gap-2">
                  <div className="mt-0.5">
                    {isRunning ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
                    ) : isPending ? (
                      <Circle className="h-3.5 w-3.5 text-muted-foreground/40" />
                    ) : failed ? (
                      <AlertCircle className="h-3.5 w-3.5 text-amber-500" />
                    ) : (
                      <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                    )}
                  </div>
                  <span className="text-xs font-medium text-foreground/90">
                    {resolveName(step.agentKey)}
                  </span>
                  <span className="text-[10px] text-muted-foreground/60">
                    {step.stepIndex + 1}/{step.stepCount}
                  </span>
                  <span className="ml-auto text-[10px] uppercase tracking-wide text-muted-foreground/60">
                    {step.status}
                  </span>
                </div>
                {step.subTask && (
                  <p className="mt-1 pl-5 text-[11px] leading-snug text-muted-foreground/80">
                    {step.subTask}
                  </p>
                )}
                {step.output && (
                  <p className="mt-1 pl-5 text-[11px] leading-snug text-muted-foreground/55 line-clamp-3 whitespace-pre-wrap">
                    {step.output}
                  </p>
                )}
              </div>
            );
          })}
      </div>
    </div>
  );
}

function formatWorkedLabel(seconds: number): string {
  if (seconds < 1) {
    return `Worked for ${Math.max(1, Math.round(seconds * 1000))} ms >`;
  }
  if (seconds > 59) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `Worked for ${minutes}m ${remainingSeconds}s >`;
  }
  return `Worked for ${Math.round(seconds)} sec >`;
}

function isRouterProviderComplete(role: RouterSettingsForm): boolean {
  const meta = PROVIDER_META[role.provider_type];
  return (
    role.model.trim().length > 0 &&
    (!meta.needsBaseUrl || role.base_url.trim().length > 0)
  );
}

function modelLabelForValue(
  modelValue: string,
): string {
  const normalized = modelValue.trim();
  if (!normalized) return "Select model";
  return normalized;
}

const ROUTER_DEFAULTS_BY_PROVIDER: Record<
  ProviderType,
  { base_url: string }
> = {
  openai_compatible: {
    base_url: "",
  },
  anthropic: {
    base_url: "",
  },
  openrouter: {
    base_url: "",
  },
};

function routerSettingsFromProfile(profile: UserProfileConfig | null | undefined): RouterSettingsForm {
  const configured = profile?.router_provider;
  const providerType = configured?.type ?? "openrouter";
  const defaults = ROUTER_DEFAULTS_BY_PROVIDER[providerType];
  return {
    provider_type: providerType,
    base_url: configured?.base_url ?? defaults.base_url,
    api_key: "",
    model: configured?.model ?? "",
    temperature:
      configured?.temperature === null || configured?.temperature === undefined
        ? "0.2"
        : String(configured.temperature),
  };
}

// ---------------------------------------------------------------------------
// SessionsPage
// ---------------------------------------------------------------------------

export function SessionsPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const qc = useQueryClient();

  const [agentKey, setAgentKey] = useState<string>("");
  const [conversationId, setConversationId] = useState<string>("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [nodeEvents, setNodeEvents] = useState<NodeEvent[]>([]);
  const [subagentSteps, setSubagentSteps] = useState<SubagentStep[]>([]);
  const [planIntro, setPlanIntro] = useState<string>("");
  const [runState, setRunState] = useState<RunState>("idle");
  const [userInput, setUserInput] = useState("");
  const [loadingSession, setLoadingSession] = useState(false);
  const [workedSeconds, setWorkedSeconds] = useState<number | null>(null);
  const [liveWorkedSeconds, setLiveWorkedSeconds] = useState(0);
  const [timerVersion, setTimerVersion] = useState(0);
  const [providerSettingsOpen, setProviderSettingsOpen] = useState(false);
  const [routerRole, setRouterRole] = useState<RouterSettingsForm>(() =>
    routerSettingsFromProfile(null),
  );

  const abortRef = useRef<AbortController | null>(null);
  const stopIntentionalRef = useRef(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const streamingAssistantIdRef = useRef<string | null>(null);
  const pendingConversationIdRef = useRef<string | null>(null);
  const runStartedAtRef = useRef<number | null>(null);
  // LangGraph Server thread ID for the current conversation (different from the
  // conversation UUID used as the URL param — see lgClient.ts for the mapping).
  const lgThreadIdRef = useRef<string | null>(null);

  // Abort any active SSE stream cleanly before the page unloads so the browser
  // loading indicator doesn't hang and the server can detect the disconnect.
  useEffect(() => {
    const handleBeforeUnload = () => {
      abortRef.current?.abort();
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, []);

  useEffect(() => {
    if (!userInput && inputRef.current) {
      inputRef.current.style.height = "";
    }
  }, [userInput]);

  const { data: agents } = useQuery({
    queryKey: ["agents"],
    queryFn: api.listAgents,
  });

  const { data: profile } = useQuery({
    queryKey: ["profile"],
    queryFn: api.getProfile,
  });

  useEffect(() => {
    if (!profile) return;
    setRouterRole(routerSettingsFromProfile(profile));
  }, [profile]);

  const saveRouterSettings = useMutation({
    mutationFn: async () => {
      const temperature = Number.parseFloat(routerRole.temperature || "0.2");
      const providerMeta = PROVIDER_META[routerRole.provider_type];
      const resolvedBaseUrl = providerMeta.needsBaseUrl
        ? routerRole.base_url.trim() || null
        : providerMeta.defaultBaseUrl;
      const nextProfile: UpdateProfileRequest = {
        display_name: profile?.display_name ?? null,
        location: profile?.location ?? null,
        timezone: profile?.timezone ?? null,
        personality: profile?.personality ?? null,
        preferences: profile?.preferences ?? {},
        router_provider: {
          type: routerRole.provider_type,
          base_url: resolvedBaseUrl,
          model: routerRole.model.trim(),
          temperature: Number.isFinite(temperature) ? temperature : 0.2,
        },
        router_api_key: routerRole.api_key.trim(),
      };
      return api.updateProfile(nextProfile);
    },
    onSuccess: (updated) => {
      qc.setQueryData(["profile"], updated);
      setRouterRole(routerSettingsFromProfile(updated));
      setProviderSettingsOpen(false);
    },
  });

  const loadConversationSnapshot = useCallback(
    async (
      targetConversationId: string,
      options?: { silentRunning?: boolean },
    ) => {
      const [stateRes, runRes] = await Promise.all([
        api.conversationSnapshot(targetConversationId),
        api.conversationRunStatus(targetConversationId),
      ]);
      const effectiveStatus =
        runRes.status !== "idle"
          ? runRes.status
          : "idle";
      const state = stateRes.state;
      const loadedAgentKey =
        typeof state.agent_key === "string" ? state.agent_key : null;
      const loadedThreadId =
        typeof stateRes.thread_id === "string" ? stateRes.thread_id : null;
      const msgs: ChatMessage[] = [];
      const chatHistory = Array.isArray(state.chat_history)
        ? (state.chat_history as PersistedChatHistoryEntry[])
        : [];
      for (const entry of chatHistory) {
        if (entry?.role === "user" && typeof entry.content === "string") {
          msgs.push({ role: "user", content: entry.content, id: msgId() });
        } else if (
          entry?.role === "assistant" &&
          typeof entry.content === "string"
        ) {
          msgs.push({ role: "assistant", content: entry.content, id: msgId() });
        }
      }
      if (effectiveStatus === "interrupted") {
        try {
          const interruptRes =
            await api.conversationInterrupt(targetConversationId);
          if (interruptRes.interrupt) {
            msgs.push({
              role: "interrupt",
              payload: interruptRes.interrupt,
              threadId: threadId ?? "",
              id: msgId(),
            });
          }
        } catch {
          // Best effort — interrupt payload unavailable
        }
      } else if (effectiveStatus === "error" && runRes.error) {
        msgs.push({ role: "error", content: runRes.error, id: msgId() });
      }
      if (msgs.length === 0)
        msgs.push({
          role: "system",
          content: `Conversation ${targetConversationId.slice(0, 8)} loaded.`,
          id: msgId(),
        });
      if (state.agent_key && typeof state.agent_key === "string")
        setAgentKey(state.agent_key);
      if (stateRes.thread_id && typeof stateRes.thread_id === "string")
        setThreadId(stateRes.thread_id);
      if (runRes.run_id) setRunId(runRes.run_id);
      runStartedAtRef.current = null;
      setMessages(msgs);
      setRunState(mapRemoteRunStatus(effectiveStatus));
      return {
        status: effectiveStatus,
        agentKey: loadedAgentKey,
        threadId: loadedThreadId,
      };
    },
    [],
  );

  const beginWorkedTimer = useCallback(() => {
    runStartedAtRef.current = Date.now();
    setLiveWorkedSeconds(0);
    setTimerVersion((current) => current + 1);
  }, []);

  const finalizeWorkedTimer = useCallback(() => {
    if (runStartedAtRef.current == null) return null;
    const elapsedSeconds = Math.max(
      0,
      (Date.now() - runStartedAtRef.current) / 1000,
    );
    runStartedAtRef.current = null;
    setLiveWorkedSeconds(0);
    setWorkedSeconds(elapsedSeconds);
    return elapsedSeconds;
  }, []);

  // Load conversation when ?conversation= param changes
  const conversationParam = searchParams.get("conversation");
  useEffect(() => {
    if (!conversationParam) {
      if (pendingConversationIdRef.current) return;

      if (
        conversationId ||
        threadId ||
        runId ||
        messages.length > 0 ||
        nodeEvents.length > 0 ||
        runState !== "idle"
      ) {
        abortRef.current?.abort();
        streamingAssistantIdRef.current = null;
        lgThreadIdRef.current = null;
        setConversationId("");
        setThreadId(null);
        setRunId(null);
        setMessages([]);
        setNodeEvents([]);
        setRunState("idle");
        setUserInput("");
        setLoadingSession(false);
        setWorkedSeconds(null);
        setLiveWorkedSeconds(0);
        setTimerVersion(0);
        runStartedAtRef.current = null;
      }
      return;
    }

    if (pendingConversationIdRef.current === conversationParam) {

      return;
    }

    if (conversationParam === conversationId) return;

    // New conversation/session — reset everything
    abortRef.current?.abort();
    streamingAssistantIdRef.current = null;
    lgThreadIdRef.current = null;
    setConversationId(conversationParam);
    setThreadId(null);
    setRunId(null);
    setMessages([]);
    setNodeEvents([]);
    setRunState("idle");
    setUserInput("");
    setLoadingSession(true);
    setWorkedSeconds(null);
    setLiveWorkedSeconds(0);
    setTimerVersion(0);
    runStartedAtRef.current = null;

    loadConversationSnapshot(conversationParam, { silentRunning: true })
      .catch(() => {
        setMessages([
          {
            role: "system",
            content: `Conversation ${conversationParam.slice(0, 8)} loaded.`,
            id: msgId(),
          },
        ]);
      })
      .finally(() => setLoadingSession(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    conversationParam,
    conversationId,
    loadConversationSnapshot,
    navigate,
    nodeEvents.length,
    messages.length,
    runId,
    runState,
    threadId,
  ]);

  useEffect(() => {
    if (!conversationId || runState !== "streaming" || abortRef.current) return;

    const interval = window.setInterval(async () => {
      try {
        const status = await api.conversationRunStatus(conversationId);
        const mapped = mapRemoteRunStatus(status.status);
        if (mapped !== "streaming") {
          await loadConversationSnapshot(conversationId);
        }
      } catch {
        // Best effort polling only.
      }
    }, 2000);

    return () => window.clearInterval(interval);
  }, [conversationId, runState, loadConversationSnapshot]);

  useEffect(() => {
    if (runState !== "streaming" || runStartedAtRef.current == null) return;

    const syncElapsedSeconds = () => {
      if (runStartedAtRef.current == null) return;
      setLiveWorkedSeconds(
        Math.max(0, Math.ceil((Date.now() - runStartedAtRef.current) / 1000)),
      );
    };

    syncElapsedSeconds();
    const interval = window.setInterval(syncElapsedSeconds, 1000);
    return () => window.clearInterval(interval);
  }, [runState, timerVersion]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);
  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);
  useEffect(() => {
    if (runState !== "streaming") return;
    scrollToBottom();
  }, [nodeEvents, runState, scrollToBottom]);

  function addMessage(msg: ChatMessageInput) {
    setMessages((prev) => [...prev, { ...msg, id: msgId() } as ChatMessage]);
  }

  function ensureStreamingAssistantMessage() {
    if (streamingAssistantIdRef.current) return streamingAssistantIdRef.current;
    const id = msgId();
    streamingAssistantIdRef.current = id;
    setMessages((prev) => [...prev, { role: "assistant", content: "", id }]);
    return id;
  }

  function appendStreamingAssistantDelta(delta: string) {
    const id = ensureStreamingAssistantMessage();
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === id && msg.role === "assistant"
          ? { ...msg, content: msg.content + delta }
          : msg,
      ),
    );
  }

  function finalizeStreamingAssistant(output: string) {
    const id = streamingAssistantIdRef.current;
    if (!id) {
      if (output) addMessage({ role: "assistant", content: output });
      return;
    }
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === id && msg.role === "assistant"
          ? { ...msg, content: msg.content || output }
          : msg,
      ),
    );
    streamingAssistantIdRef.current = null;
  }

  function markNodeDone(node: string) {
    setNodeEvents((prev) =>
      prev.map((ev) =>
        ev.node === node && ev.status === "running"
          ? { ...ev, status: "done" }
          : ev,
      ),
    );
  }

  const handleEvent = useCallback(
    (e: StreamEvent) => {
      if (e.event === "start") {
        const tid = e.data.thread_id;
        setThreadId(tid);
        if (e.data.run_id) {
          setRunId(e.data.run_id);
        }
        if (e.data.agent_key) {
          setAgentKey(e.data.agent_key);
        }
        pendingConversationIdRef.current = null;
        setNodeEvents([]);
        ensureStreamingAssistantMessage();
        beginWorkedTimer();
      } else if (e.event === "message_start") {
        ensureStreamingAssistantMessage();
      } else if (e.event === "message_delta") {
        appendStreamingAssistantDelta(e.data.delta);
      } else if (e.event === "node") {
        const tokens = e.data.tokens ?? undefined;
        setNodeEvents((prev) => {
          const last = prev[prev.length - 1];
          if (last?.status === "running") {
            return [
              ...prev.slice(0, -1),
              { ...last, status: "done" as const },
              {
                node: e.data.node,
                phase: e.data.phase,
                ts: Date.now(),
                status: "running" as const,
                tokens,
              },
            ];
          }
          return [
            ...prev,
            {
              node: e.data.node,
              phase: e.data.phase,
              ts: Date.now(),
              status: "running" as const,
              tokens,
            },
          ];
        });
      } else if (e.event === "plan") {
        // The router (manager) announces what it will do before any worker runs.
        // Pre-populate every step as pending so the user sees the full plan upfront.
        setPlanIntro(e.data.intro || "");
        setSubagentSteps(
          e.data.steps.map((s) => ({
            agentKey: s.agent_key,
            subTask: s.sub_task,
            stepIndex: s.step_index,
            stepCount: e.data.step_count,
            status: "pending" as const,
            output: "",
          })),
        );
      } else if (e.event === "subagent") {
        // The router (manager) is dispatching a worker. Stay in this chat —
        // just track which sub-agent is working and its status.
        const d = e.data;
        setSubagentSteps((prev) => {
          const existing = prev.findIndex((s) => s.stepIndex === d.step_index);
          const next: SubagentStep = {
            agentKey: d.agent_key,
            subTask: d.sub_task,
            stepIndex: d.step_index,
            stepCount: d.step_count,
            status: d.status,
            output: existing >= 0 ? prev[existing].output : "",
          };
          if (existing >= 0) {
            const copy = [...prev];
            copy[existing] = { ...copy[existing], ...next, output: copy[existing].output };
            return copy;
          }
          return [...prev, next];
        });
      } else if (e.event === "subagent_delta") {
        const d = e.data;
        setSubagentSteps((prev) =>
          prev.map((s) =>
            s.stepIndex === d.step_index
              ? { ...s, output: s.output + d.delta }
              : s,
          ),
        );
      } else if (e.event === "interrupt") {
        markNodeDone(nodeEvents[nodeEvents.length - 1]?.node ?? "");
        setRunState("interrupted");
        finalizeWorkedTimer();
        setThreadId(e.data.thread_id);
        abortRef.current = null;
        addMessage({
          role: "interrupt",
          payload: e.data.interrupt,
          threadId: e.data.thread_id,
        });
      } else if (e.event === "completed") {
        setNodeEvents((prev) => prev.map((ev) => ({ ...ev, status: "done" })));
        setRunState("completed");
        finalizeWorkedTimer();
        abortRef.current = null;
        finalizeStreamingAssistant(e.data.output);
        qc.invalidateQueries({ queryKey: ["threads"] });
      } else if (e.event === "cancelled") {
        setNodeEvents((prev) => prev.map((ev) => ({ ...ev, status: "done" })));
        setRunState("cancelled");
        finalizeWorkedTimer();
        streamingAssistantIdRef.current = null;
        abortRef.current = null;
        setNodeEvents([]);
      } else if (e.event === "error") {
        setRunState("error");
        finalizeWorkedTimer();
        streamingAssistantIdRef.current = null;
        abortRef.current = null;
        addMessage({ role: "error", content: e.data.message });
        setNodeEvents([]);
      }
    },
    [
      beginWorkedTimer,
      conversationId,
      finalizeWorkedTimer,
      nodeEvents,
      qc,
    ],
  );

  async function startRun(
    input: string,
    conversationIdOverride?: string,
  ) {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setRunState("streaming");
    setNodeEvents([]);
    setSubagentSteps([]);
    setPlanIntro("");

    const effectiveConversationId = conversationIdOverride || conversationId || undefined;

    try {
      // ── LangGraph Server path (chat/clarify intents) ──────────────────────
      // Ensure we have a LangGraph thread for this conversation.  On the very
      // first message for a conversation the thread doesn't exist yet; create it
      // and store the mapping so page reloads can look it up.
      if (!lgThreadIdRef.current && effectiveConversationId) {
        const lgThreadId = await lgCreateThread(effectiveConversationId);
        lgThreadIdRef.current = lgThreadId;
        persistThreadMapping(effectiveConversationId, lgThreadId);
      }

      if (lgThreadIdRef.current) {
        await streamLangGraph(lgThreadIdRef.current, input, handleEvent, ctrl.signal);
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        streamingAssistantIdRef.current = null;
        pendingConversationIdRef.current = null;
        abortRef.current = null;
        addMessage({
          role: "error",
          content: (err as ApiError).message ?? String(err),
        });
        finalizeWorkedTimer();
        setRunState("error");
      } else {
        streamingAssistantIdRef.current = null;
        pendingConversationIdRef.current = null;
        abortRef.current = null;
        finalizeWorkedTimer();
        if (stopIntentionalRef.current) {
          stopIntentionalRef.current = false;
        } else {
          setRunState("idle");
        }
      }
    }
  }

  function handleSend() {
    if (!userInput.trim() || runState === "streaming") return;
    const text = userInput.trim();
    if (!conversationId) {
      const nextConversationId = crypto.randomUUID();
      pendingConversationIdRef.current = nextConversationId;
      setConversationId(nextConversationId);
      const nextParams = new URLSearchParams(searchParams);
      nextParams.set("conversation", nextConversationId);
      nextParams.delete("agent");
      navigate(`/sessions?${nextParams.toString()}`, { replace: true });
      setUserInput("");
      addMessage({ role: "user", content: text });
      startRun(text, nextConversationId);
      return;
    }
    setUserInput("");
    addMessage({ role: "user", content: text });
    startRun(text);
  }

  async function handleStop() {
    stopIntentionalRef.current = true;
    abortRef.current?.abort();
    abortRef.current = null;
    streamingAssistantIdRef.current = null;
    finalizeWorkedTimer();
    setRunState("cancelled");
    setNodeEvents((prev) => prev.map((ev) => ({ ...ev, status: "done" })));
    const lgThreadId = lgThreadIdRef.current ?? (conversationId ? loadThreadMapping(conversationId) : null);
    if (lgThreadId && runId) {
      try {
        await lgCancelRun(lgThreadId, runId);
      } catch {
        /* best effort */
      }
    }
  }

  const streaming = runState === "streaming";
  const taskRunning = streaming;
  const streamingEmptyStateLabel =
    streaming && nodeEvents.length === 0 ? "Starting..." : null;
  const displayedWorkedSeconds =
      streaming
        ? liveWorkedSeconds
        : workedSeconds;
  const showWorkedTimer = agentKey !== "router" && !!displayedWorkedSeconds;
  const providerMeta = PROVIDER_META[routerRole.provider_type];
  const currentModelLabel = modelLabelForValue(routerRole.model);
  const canSend =
    !!userInput.trim() &&
    isRouterProviderComplete(routerRole) &&
    !streaming &&
    runState !== "interrupted" &&
    runState !== "cancelled";

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Chat pane + node rail */}
      <div className="flex flex-1 min-h-0">
        {/* Chat pane */}
        <div className="flex flex-col flex-1 min-w-0 border-r bg-zinc-50/70 dark:bg-transparent">
          {/* Agent selector bar */}
          <div className="flex h-14 items-center gap-3 px-4 shrink-0">
            <Bot className="h-4 w-4 shrink-0 text-muted-foreground" />
            {conversationId ? (
              <>
                <span className="text-sm font-medium">
                  {agents?.find((a) => a.agent_key === agentKey)?.name ??
                    (agentKey || "Routed chat")}
                </span>
                <Badge variant="outline" className="ml-auto font-mono text-xs">
                  {conversationId.slice(0, 8)}
                </Badge>
              </>
            ) : (
              <span className="text-sm text-muted-foreground">
                Routed chat
              </span>
            )}
          </div>

          {/* Messages */}
          <div className="flex-1 min-h-0 overflow-y-auto px-4 sm:px-8 lg:px-16 [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border [&::-webkit-scrollbar-track]:bg-transparent">
            {messages.length === 0 && !streaming && !loadingSession ? (
              <div className="flex min-h-full items-center justify-center py-4 text-center">
                <div className="flex flex-col items-center">
                  <img src="/logo.png" alt="solidcue" className="mb-3 h-16 w-16 opacity-30" />
                  <p className="text-sm font-medium text-muted-foreground">No messages yet</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Type a message below to start.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-4 py-4">
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
                      <div className="min-w-0 max-w-[80%] overflow-hidden px-4 py-2.5 text-foreground/80">
                        {msg.content.trim() ? (
                          <MarkdownContent content={msg.content} />
                        ) : streaming && streamingAssistantIdRef.current === msg.id ? (
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            <span>Thinking...</span>
                          </div>
                        ) : (
                          <MarkdownContent content={msg.content} />
                        )}
                      </div>
                    </div>
                  )}
                  {msg.role === "interrupt" && (
                    <ApprovalCard
                      payload={msg.payload}
                      onSubmit={() => {}}
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
                      <p className="text-center text-xs text-muted-foreground">
                        {msg.content}
                      </p>
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

                {nodeEvents.length > 0 && (
                  <StepHistory events={nodeEvents} />
                )}

                {(subagentSteps.length > 0 || planIntro) && (
                  <SubagentActivity
                    steps={subagentSteps}
                    intro={planIntro}
                    resolveName={(key) =>
                      agents?.find((a) => a.agent_key === key)?.name ?? key
                    }
                  />
                )}

                {showWorkedTimer && (
                  <div className="px-1">
                    <div className="mb-2 h-px w-full bg-border/60" />
                    <p className="text-[11px] text-muted-foreground/70">
                      {formatWorkedLabel(displayedWorkedSeconds)}
                    </p>
                  </div>
                )}

                {streamingEmptyStateLabel && (
                  <div className="px-1">
                    <div className="flex items-center gap-1.5">
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                      <span className="text-[11px] text-muted-foreground/75">
                        {streamingEmptyStateLabel}
                      </span>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input */}
          <div className="shrink-0 px-4 sm:px-8 lg:px-16 pb-4 pt-2">
            <div className="relative rounded-[26px] border border-border/80 bg-card/95 px-3.5 py-2.5 shadow-sm transition-all focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/30">
              <div className="flex items-start gap-2">
                <textarea
                  ref={inputRef}
                  value={userInput}
                  onChange={(e) => {
                    setUserInput(e.target.value);
                    const el = e.target;
                    el.style.height = "auto";
                    el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
                  }}
                  placeholder={taskRunning ? "Task is running..." : "Type your message..."}
                  rows={1}
                  disabled={taskRunning}
                  className="flex-1 resize-none bg-transparent py-1 text-[15px] text-foreground placeholder:text-muted-foreground/80 focus:outline-none disabled:opacity-50 leading-relaxed max-h-[150px] overflow-y-auto"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      if (canSend) handleSend();
                    }
                  }}
                />
              </div>
              <div className="mt-3 flex items-center justify-end gap-2">
                <div />
                <DropdownMenu>
                  <DropdownMenuTrigger asChild disabled={taskRunning}>
                    <button
                      type="button"
                      className="inline-flex h-9 items-center gap-2 rounded-xl px-2.5 text-xs text-foreground/85 transition-colors hover:bg-muted/40 disabled:opacity-50"
                    >
                      <span className="max-w-[180px] truncate">{currentModelLabel}</span>
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-72 rounded-2xl p-2">
                    {routerRole.model.trim() ? (
                      <>
                        <DropdownMenuItem
                          disabled
                          className="rounded-xl px-3 py-2.5 opacity-100 focus:bg-transparent focus:text-foreground"
                        >
                          <span className="flex-1">{routerRole.model.trim()}</span>
                          <Check className="h-4 w-4 text-foreground" />
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                      </>
                    ) : null}
                    <DropdownMenuItem
                      onSelect={() => setProviderSettingsOpen(true)}
                      className="rounded-xl px-3 py-2.5"
                    >
                      <Settings2 className="h-4 w-4" />
                      <span>Provider settings</span>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
                {taskRunning ? (
                  <button
                    type="button"
                    onClick={handleStop}
                    className="flex h-8.5 w-8.5 shrink-0 items-center justify-center rounded-full bg-muted hover:text-destructive transition-colors"
                  >
                    <Square className="h-3.5 w-3.5 fill-current" />
                  </button>
                ) : (
                  <button
                    type="button"
                    disabled={!canSend}
                    onClick={handleSend}
                    className="flex h-8.5 w-8.5 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground transition-all hover:opacity-90 disabled:bg-muted disabled:text-muted-foreground active:scale-95"
                  >
                    <Send className="h-4 w-4" />
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>

      </div>

      <Dialog open={providerSettingsOpen} onOpenChange={setProviderSettingsOpen}>
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>Provider Settings</DialogTitle>
            <DialogDescription>
              Configure the router model and provider connection used before agent handoff.
            </DialogDescription>
          </DialogHeader>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Provider</label>
              <Select
                value={routerRole.provider_type}
                onValueChange={(value) =>
                  setRouterRole((current) => {
                    const nextProvider = value as ProviderType;
                    return {
                      ...current,
                      provider_type: nextProvider,
                      base_url:
                        nextProvider === "openai_compatible" ? current.base_url : "",
                      api_key: "",
                    };
                  })
                }
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(PROVIDER_META).map(([key, meta]) => (
                    <SelectItem key={key} value={key}>
                      {meta.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium">Model</label>
              <Input
                value={routerRole.model}
                onChange={(e) =>
                  setRouterRole((current) => ({
                    ...current,
                    model: e.target.value,
                  }))
                }
                placeholder="model-name"
              />
            </div>

            {providerMeta.needsBaseUrl && (
              <div className="space-y-1.5 sm:col-span-2">
                <label className="text-sm font-medium">Base URL</label>
                <Input
                  value={routerRole.base_url}
                  onChange={(e) =>
                    setRouterRole((current) => ({
                      ...current,
                      base_url: e.target.value,
                    }))
                  }
                  placeholder="https://api.example.com/v1"
                />
              </div>
            )}

            <div className="space-y-1.5">
              <label className="text-sm font-medium">Temperature</label>
              <Input
                type="number"
                step="0.1"
                min="0"
                max="2"
                value={routerRole.temperature}
                onChange={(e) =>
                  setRouterRole((current) => ({
                    ...current,
                    temperature: e.target.value,
                  }))
                }
                placeholder="0.2"
              />
            </div>

            <div className="space-y-1.5 sm:col-span-2">
              <label className="text-sm font-medium">API Key</label>
              <Input
                type="password"
                value={routerRole.api_key}
                onChange={(e) =>
                  setRouterRole((current) => ({
                    ...current,
                    api_key: e.target.value,
                  }))
                }
                placeholder="Paste API key"
                autoComplete="off"
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setProviderSettingsOpen(false)}>
              Close
            </Button>
            <Button
              onClick={() => saveRouterSettings.mutate()}
              disabled={!isRouterProviderComplete(routerRole) || saveRouterSettings.isPending}
            >
              {saveRouterSettings.isPending ? "Saving..." : "Save"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
