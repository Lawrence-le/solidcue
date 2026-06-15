import { useState, useMemo } from "react"
import { NavLink, useNavigate, useSearchParams } from "react-router-dom"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Bot,
  Check,
  Loader2,
  MoreVertical,
  Moon,
  Plus,
  PlugZap,
  Sun,
  Trash2,
  User,
  Wrench,
} from "lucide-react"
import { toast } from "sonner"
import { api, ApiError } from "@/lib/api"
import { lgDeleteThread, lgListThreads } from "@/lib/lgClient"
import type { ThreadSummary } from "@/lib/types"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { useTheme } from "@/components/theme-provider"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

const NAV = [
  { to: "/sessions", label: "New", icon: Plus },
  { to: "/agents", label: "Agents", icon: Bot },
  { to: "/mcp", label: "MCP", icon: PlugZap },
  { to: "/tools", label: "Tools", icon: Wrench },
  { to: "/profile", label: "Profile", icon: User },
]

function sessionIdForThread(thread: ThreadSummary | null): string {
  if (!thread) return ""
  return thread.conversation_id || thread.thread_id
}

export function Sidebar() {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const { resolved, toggle } = useTheme()
  const [searchParams] = useSearchParams()
  const activeConversation = searchParams.get("conversation") ?? searchParams.get("thread")
  const [agentFilter, setAgentFilter] = useState("all")
  const [threadToDelete, setThreadToDelete] = useState<ThreadSummary | null>(null)
  const [selectionMode, setSelectionMode] = useState(false)
  const [selectedConversationIds, setSelectedConversationIds] = useState<string[]>([])

  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15000,
    retry: false,
  })

  const { data: threads, isLoading } = useQuery({
    queryKey: ["threads"],
    queryFn: async () => {
      try {
        return await lgListThreads()
      } catch (error) {
        if (error instanceof ApiError && error.status === 502) {
          return []
        }
        throw error
      }
    },
    refetchInterval: 10_000,
    retry: false,
  })

  const agentKeys = useMemo(() => {
    const keys = new Set(threads?.map((t) => t.agent_key ?? "unknown"))
    return Array.from(keys).sort()
  }, [threads])

  const visibleThreads = useMemo(() => {
    if (agentFilter === "all") return threads ?? []
    return (threads ?? []).filter((t) => (t.agent_key ?? "unknown") === agentFilter)
  }, [threads, agentFilter])

  const selectedThreads = useMemo(
    () => visibleThreads.filter((thread) => selectedConversationIds.includes(sessionIdForThread(thread))),
    [visibleThreads, selectedConversationIds],
  )
  const allVisibleSelected =
    visibleThreads.length > 0 && visibleThreads.every((thread) => selectedConversationIds.includes(sessionIdForThread(thread)))
  const ok = health.isSuccess && health.data?.status === "ok"

  const deleteThread = useMutation({
    mutationFn: async (conversationId: string) => {
      const deleted = await lgDeleteThread(conversationId)
      if (!deleted) throw new Error("Could not delete session")
    },
    onSuccess: (_, deletedConversationId) => {
      toast.success("Session deleted")
      qc.invalidateQueries({ queryKey: ["threads"] })
      if (activeConversation === deletedConversationId) {
        navigate("/sessions")
      }
      setThreadToDelete(null)
    },
    onError: (error: ApiError) => toast.error(error.message),
  })

  const deleteThreads = useMutation({
    mutationFn: async (conversationIds: string[]) => {
      const results = await Promise.all(conversationIds.map((conversationId) => lgDeleteThread(conversationId)))
      if (results.some((deleted) => !deleted)) {
        throw new Error("Could not delete one or more sessions")
      }
      return conversationIds
    },
    onSuccess: (deletedConversationIds) => {
      toast.success(
        deletedConversationIds.length === 1 ? "Session deleted" : `${deletedConversationIds.length} sessions deleted`,
      )
      qc.invalidateQueries({ queryKey: ["threads"] })
      if (
        activeConversation &&
        deletedConversationIds.includes(activeConversation)
      ) {
        navigate("/sessions")
      }
      setSelectedConversationIds([])
      setSelectionMode(false)
      setThreadToDelete(null)
    },
    onError: (error: ApiError) => toast.error(error.message),
  })

  function toggleThreadSelected(conversationId: string) {
    setSelectedConversationIds((prev) =>
      prev.includes(conversationId) ? prev.filter((id) => id !== conversationId) : [...prev, conversationId],
    )
  }

  function exitSelectionMode() {
    setSelectionMode(false)
    setSelectedConversationIds([])
  }

  function toggleSelectAll() {
    if (allVisibleSelected) {
      setSelectedConversationIds([])
      return
    }
    setSelectedConversationIds(visibleThreads.map((thread) => sessionIdForThread(thread)))
  }

  return (
    <aside className="flex h-full w-full flex-col">
      {/* Logo */}
      <div className="flex h-14 shrink-0 items-center gap-2 border-b border-border px-4">
        <img src="/logo.png" alt="solidcue" className="h-7 w-7 rounded-md" />
        <span className="min-w-0 flex-1 truncate font-semibold tracking-tight">solidcue studio</span>
      </div>

      {/* Nav items */}
      <nav className="flex shrink-0 flex-col gap-0.5 p-1.5">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground",
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Conversation list */}
      <div className="mx-2 border-t border-border/50" />

      {agentKeys.length > 1 && (
        <div className="mt-1.5 shrink-0 px-3 pb-2 flex items-center gap-2">
          <span className="text-xs text-muted-foreground shrink-0">Agent</span>
          <div className="relative flex-1">
            <select
              value={agentFilter}
              onChange={(e) => setAgentFilter(e.target.value)}
              className="w-full appearance-none rounded-md border border-border bg-background pl-2 pr-7 py-1 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            >
              <option value="all">All</option>
              {agentKeys.map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
            <svg className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </div>
          {!selectionMode && (
            <Button
              type="button"
              size="xs"
              variant="ghost"
              onClick={() => setSelectionMode(true)}
              className="shrink-0 text-xs text-muted-foreground"
              disabled={!visibleThreads.length}
            >
              Select
            </Button>
          )}
        </div>
      )}

      {agentKeys.length <= 1 && !selectionMode && (
        <div className="mt-1.5 shrink-0 px-3 pb-2 flex justify-end">
          <Button
            type="button"
            size="xs"
            variant="ghost"
            onClick={() => setSelectionMode(true)}
            className="text-xs text-muted-foreground"
            disabled={!visibleThreads.length}
          >
            Select
          </Button>
        </div>
      )}

      {selectionMode && (
        <div
          className={cn(
            "mt-1.5 shrink-0 px-3 pb-2 flex items-center justify-end gap-1",
          )}
        >
          <Button
            type="button"
            size="xs"
            variant="ghost"
            onClick={toggleSelectAll}
            className="text-xs text-muted-foreground"
            disabled={!visibleThreads.length}
          >
            {allVisibleSelected ? "Clear all" : "Select all"}
          </Button>
          <Button
            type="button"
            size="xs"
            variant="ghost"
            onClick={exitSelectionMode}
            className="text-xs text-muted-foreground"
          >
            Cancel
          </Button>
          <Button
            type="button"
            size="xs"
            variant="ghost"
            disabled={selectedConversationIds.length === 0}
            onClick={() =>
              setThreadToDelete(
                selectedThreads.length === 1
                  ? selectedThreads[0]
                  : ({ conversation_id: "__multi__", thread_id: "__multi__", agent_key: null, step_count: 0 } as ThreadSummary),
              )
            }
            className="text-xs text-destructive hover:text-destructive"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Delete
          </Button>
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-y-auto [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border [&::-webkit-scrollbar-track]:bg-transparent">
        {isLoading && (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          </div>
        )}
        {!isLoading && visibleThreads.length === 0 && (
          <p className="px-3 py-3 text-xs text-muted-foreground">No sessions yet.</p>
        )}
        {visibleThreads.map((t) => (
          <div
            key={sessionIdForThread(t)}
            className={cn(
              "mx-2 flex items-center gap-2 rounded-md border-b border-border/40 px-2.5 py-1 hover:bg-accent/60 transition-colors",
              activeConversation === sessionIdForThread(t) && "bg-primary/10",
            )}
          >
            {selectionMode && (
              <button
                type="button"
                onClick={() => toggleThreadSelected(sessionIdForThread(t))}
                className={cn(
                  "flex h-6 w-6 shrink-0 items-center justify-center rounded-full border transition-colors",
                  selectedConversationIds.includes(sessionIdForThread(t))
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-background",
                )}
                aria-label={`Select session ${sessionIdForThread(t).slice(0, 8)}`}
              >
                {selectedConversationIds.includes(sessionIdForThread(t)) ? <Check className="h-3 w-3" /> : null}
              </button>
            )}
            <button
              type="button"
              onClick={() =>
                selectionMode ? toggleThreadSelected(sessionIdForThread(t)) : navigate(`/sessions?conversation=${sessionIdForThread(t)}`)
              }
              className="min-w-0 flex-1 text-left"
            >
              <p className="text-xs truncate text-foreground">
                <span className="font-mono font-medium">{sessionIdForThread(t).slice(0, 8)}</span>
                <span className="text-muted-foreground/60"> · {t.agent_key ?? "unknown"} · {t.step_count}s</span>
              </p>
            </button>
            {!selectionMode && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    className="shrink-0 text-muted-foreground hover:text-foreground"
                  >
                    <MoreVertical className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem variant="destructive" onClick={() => setThreadToDelete(t)}>
                    <Trash2 className="h-4 w-4" />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
          </div>
        ))}
      </div>

      <div className="shrink-0 border-t border-border/50 px-4 py-3">
        <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "h-2 w-2 rounded-full",
                health.isLoading ? "bg-warning" : ok ? "bg-success" : "bg-destructive",
              )}
            />
            <span>{health.isLoading ? "connecting" : ok ? "API ok" : "API down"}</span>
          </div>
          <Button variant="ghost" size="icon" onClick={toggle} aria-label="Toggle theme">
            {resolved === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>
      </div>

      <Dialog open={!!threadToDelete} onOpenChange={(open) => !open && setThreadToDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{threadToDelete?.thread_id === "__multi__" ? "Delete Sessions" : "Delete Session"}</DialogTitle>
            <DialogDescription>
              {threadToDelete?.thread_id === "__multi__"
                ? `Delete ${selectedConversationIds.length} selected sessions and their saved messages from the checkpoint store. This cannot be undone.`
                : "Delete this session and its saved messages from the checkpoint store. This cannot be undone."}
            </DialogDescription>
          </DialogHeader>
          {threadToDelete && threadToDelete.thread_id !== "__multi__" && (
            <div className="rounded-md border border-border bg-muted/30 px-3 py-2 text-sm">
              <span className="font-mono">{sessionIdForThread(threadToDelete).slice(0, 8)}</span>
              <span className="text-muted-foreground"> · {threadToDelete?.agent_key ?? "unknown"}</span>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setThreadToDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              disabled={!threadToDelete || deleteThread.isPending || deleteThreads.isPending}
              onClick={() => {
                if (!threadToDelete) return
                if (threadToDelete.thread_id === "__multi__") {
                  deleteThreads.mutate(selectedConversationIds)
                  return
                }
                deleteThread.mutate(sessionIdForThread(threadToDelete))
              }}
            >
              {deleteThread.isPending || deleteThreads.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </aside>
  )
}
