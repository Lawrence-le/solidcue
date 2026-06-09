import { useState, useMemo } from "react"
import { NavLink, useNavigate, useSearchParams } from "react-router-dom"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Bot, Check, Loader2, MoreVertical, Plus, PlugZap, Trash2, User, Wrench } from "lucide-react"
import { toast } from "sonner"
import { api, ApiError } from "@/lib/api"
import type { ThreadSummary } from "@/lib/types"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
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
  { to: "/agents", label: "Agents", icon: Bot },
  { to: "/mcp", label: "MCP", icon: PlugZap },
  { to: "/tools", label: "Tools", icon: Wrench },
  { to: "/profile", label: "Profile", icon: User },
]

export function Sidebar() {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [searchParams] = useSearchParams()
  const activeThread = searchParams.get("thread")
  const [agentFilter, setAgentFilter] = useState("all")
  const [threadToDelete, setThreadToDelete] = useState<ThreadSummary | null>(null)
  const [selectionMode, setSelectionMode] = useState(false)
  const [selectedThreadIds, setSelectedThreadIds] = useState<string[]>([])

  const { data: threads, isLoading } = useQuery({
    queryKey: ["threads"],
    queryFn: () => api.listThreads(),
    refetchInterval: 10_000,
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
    () => visibleThreads.filter((thread) => selectedThreadIds.includes(thread.thread_id)),
    [visibleThreads, selectedThreadIds],
  )
  const allVisibleSelected =
    visibleThreads.length > 0 && visibleThreads.every((thread) => selectedThreadIds.includes(thread.thread_id))

  const deleteThread = useMutation({
    mutationFn: (threadId: string) => api.deleteThread(threadId),
    onSuccess: (_, deletedThreadId) => {
      toast.success("Session deleted")
      qc.invalidateQueries({ queryKey: ["threads"] })
      if (activeThread === deletedThreadId) {
        navigate("/sessions")
      }
      setThreadToDelete(null)
    },
    onError: (error: ApiError) => toast.error(error.message),
  })

  const deleteThreads = useMutation({
    mutationFn: async (threadIds: string[]) => {
      await Promise.all(threadIds.map((threadId) => api.deleteThread(threadId)))
      return threadIds
    },
    onSuccess: (deletedThreadIds) => {
      toast.success(
        deletedThreadIds.length === 1 ? "Session deleted" : `${deletedThreadIds.length} sessions deleted`,
      )
      qc.invalidateQueries({ queryKey: ["threads"] })
      if (activeThread && deletedThreadIds.includes(activeThread)) {
        navigate("/sessions")
      }
      setSelectedThreadIds([])
      setSelectionMode(false)
      setThreadToDelete(null)
    },
    onError: (error: ApiError) => toast.error(error.message),
  })

  function toggleThreadSelected(threadId: string) {
    setSelectedThreadIds((prev) =>
      prev.includes(threadId) ? prev.filter((id) => id !== threadId) : [...prev, threadId],
    )
  }

  function exitSelectionMode() {
    setSelectionMode(false)
    setSelectedThreadIds([])
  }

  function toggleSelectAll() {
    if (allVisibleSelected) {
      setSelectedThreadIds([])
      return
    }
    setSelectedThreadIds(visibleThreads.map((thread) => thread.thread_id))
  }

  return (
    <aside className="flex h-full w-full flex-col">
      {/* Logo */}
      <div className="flex h-14 shrink-0 items-center gap-2 border-b border-border px-4">
        <img src="/logo.png" alt="solidcue" className="h-7 w-7 rounded-md" />
        <span className="min-w-0 flex-1 truncate font-semibold tracking-tight">solidcue studio</span>
      </div>

      {/* Nav items */}
      <nav className="flex shrink-0 flex-col gap-1 p-2">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
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

      <div className="flex items-center justify-between px-3 py-2 shrink-0 gap-2">
        <span className="min-w-0 truncate text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Conversations
        </span>
        <button
          type="button"
          onClick={() => navigate("/sessions")}
          className="flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          title="New session"
        >
          <Plus className="h-3.5 w-3.5" />
          New
        </button>
      </div>

      {agentKeys.length > 1 && (
        <div className="shrink-0 px-3 pb-2 flex items-center gap-2">
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
        <div className="shrink-0 px-3 pb-2 flex justify-end">
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
        <div className="shrink-0 px-3 pb-2">
          <div className="flex flex-wrap items-center justify-end gap-1">
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
              disabled={selectedThreadIds.length === 0}
              onClick={() =>
                setThreadToDelete(
                  selectedThreads.length === 1
                    ? selectedThreads[0]
                    : ({ thread_id: "__multi__", agent_key: null, step_count: 0 } as ThreadSummary),
                )
              }
              className="text-xs text-destructive hover:text-destructive"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Delete
            </Button>
          </div>
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
            key={t.thread_id}
            className={cn(
              "mx-2 flex items-center gap-2 rounded-md border-b border-border/40 px-3 py-2 hover:bg-accent/60 transition-colors",
              activeThread === t.thread_id && "bg-primary/10",
            )}
          >
            {selectionMode && (
              <button
                type="button"
                onClick={() => toggleThreadSelected(t.thread_id)}
                className={cn(
                  "flex h-4 w-4 shrink-0 items-center justify-center rounded-full border transition-colors",
                  selectedThreadIds.includes(t.thread_id)
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-background",
                )}
                aria-label={`Select session ${t.thread_id.slice(0, 8)}`}
              >
                {selectedThreadIds.includes(t.thread_id) ? <Check className="h-3 w-3" /> : null}
              </button>
            )}
            <button
              type="button"
              onClick={() =>
                selectionMode ? toggleThreadSelected(t.thread_id) : navigate(`/sessions?thread=${t.thread_id}`)
              }
              className="min-w-0 flex-1 text-left"
            >
              <p className="text-xs truncate text-foreground">
                <span className="font-mono font-medium">{t.thread_id.slice(0, 8)}</span>
                <span className="text-muted-foreground/60"> · {t.agent_key ?? "unknown"} · {t.step_count}s</span>
              </p>
            </button>
            {!selectionMode && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 shrink-0 text-muted-foreground hover:text-foreground"
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

      <Dialog open={!!threadToDelete} onOpenChange={(open) => !open && setThreadToDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{threadToDelete?.thread_id === "__multi__" ? "Delete Sessions" : "Delete Session"}</DialogTitle>
            <DialogDescription>
              {threadToDelete?.thread_id === "__multi__"
                ? `Delete ${selectedThreadIds.length} selected sessions and their saved messages from the checkpoint store. This cannot be undone.`
                : "Delete this session and its saved messages from the checkpoint store. This cannot be undone."}
            </DialogDescription>
          </DialogHeader>
          {threadToDelete?.thread_id !== "__multi__" && (
            <div className="rounded-md border border-border bg-muted/30 px-3 py-2 text-sm">
              <span className="font-mono">{threadToDelete?.thread_id.slice(0, 8)}</span>
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
                  deleteThreads.mutate(selectedThreadIds)
                  return
                }
                deleteThread.mutate(threadToDelete.thread_id)
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
