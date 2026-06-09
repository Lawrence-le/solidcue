import { useState, useMemo } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Loader2, Plus, Wrench } from "lucide-react"
import { toast } from "sonner"
import { api, ApiError } from "@/lib/api"
import type { APIToolConfig, MCPAuthConfig, MCPServerConfig, MCPToolConfig, ToolConfig } from "@/lib/types"
import type { ApprovalMode, ApprovalRisk, ToolType } from "@/lib/types"
import { normalizeKey } from "@/lib/agent-config"
import { AuthBuilder, authValid, emptyAuth } from "@/components/AuthBuilder"
import { ToolCard } from "@/components/ToolCard"
import { PageHeader } from "@/components/PageHeader"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"

// ---------------------------------------------------------------------------
// Shared approval fields reused by both create and edit
// ---------------------------------------------------------------------------

function ApprovalFields({
  mode,
  risk,
  prompt,
  onChange,
}: {
  mode: ApprovalMode
  risk: ApprovalRisk
  prompt: string
  onChange: (patch: { mode?: ApprovalMode; risk?: ApprovalRisk; prompt?: string }) => void
}) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <Label>Approval mode</Label>
          <Select value={mode} onValueChange={(v) => onChange({ mode: v as ApprovalMode })}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="never">Never (auto-approve)</SelectItem>
              <SelectItem value="always">Always ask</SelectItem>
              <SelectItem value="conditional">Conditional</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label>Risk level</Label>
          <Select value={risk} onValueChange={(v) => onChange({ risk: v as ApprovalRisk })}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="low">Low</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="high">High</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      {mode === "conditional" && (
        <div className="space-y-1.5">
          <Label>Approval prompt</Label>
          <Textarea
            value={prompt}
            onChange={(e) => onChange({ prompt: e.target.value })}
            placeholder="Describe when this tool needs approval…"
            rows={2}
          />
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Create tool modal (tabbed: mcp / api / rag)
// ---------------------------------------------------------------------------


function CreateToolModal({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
}) {
  const qc = useQueryClient()
  const [tab, setTab] = useState<ToolType>("mcp")
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [mode, setMode] = useState<ApprovalMode>("never")
  const [risk, setRisk] = useState<ApprovalRisk>("low")
  const [approvalPrompt, setApprovalPrompt] = useState("")

  // mcp
  const [mcpServerKey, setMcpServerKey] = useState("")
  const [mcpToolName, setMcpToolName] = useState("")

  // api
  const [apiBaseUrl, setApiBaseUrl] = useState("")
  const [apiMethod, setApiMethod] = useState<"GET" | "POST">("GET")
  const [apiHeadersRaw, setApiHeadersRaw] = useState("")
  const [apiAuth, setApiAuth] = useState<MCPAuthConfig>(emptyAuth())

  // rag (description only, no extra config needed)

  const { data: mcpServers } = useQuery<MCPServerConfig[]>({
    queryKey: ["mcp-servers"],
    queryFn: api.listMcpServers,
    enabled: tab === "mcp",
  })

  const toolKey = normalizeKey(name)

  function parseHeaders(): Record<string, string> {
    const result: Record<string, string> = {}
    for (const line of apiHeadersRaw.split("\n")) {
      const idx = line.indexOf(":")
      if (idx === -1) continue
      result[line.slice(0, idx).trim()] = line.slice(idx + 1).trim()
    }
    return result
  }

  const isValid = (() => {
    if (!name.trim() || !toolKey) return false
    if (tab === "mcp") return !!mcpServerKey && !!mcpToolName.trim()
    if (tab === "api") return !!apiBaseUrl.trim() && authValid(apiAuth)
    return true // rag
  })()

  const create = useMutation({
    mutationFn: () => {
      const base = {
        name: name.trim(),
        tool_key: toolKey,
        description: description.trim(),
        approval_mode: mode,
        approval_risk: risk,
        approval_prompt: mode === "conditional" ? approvalPrompt.trim() || null : null,
      }
      if (tab === "mcp") {
        const mcp: MCPToolConfig = {
          server_key: mcpServerKey,
          tool_name: mcpToolName.trim(),
          input_schema: null,
        }
        return api.createMcpTool({ ...base, mcp })
      }
      if (tab === "api") {
        const apiCfg: APIToolConfig = {
          base_url: apiBaseUrl.trim(),
          method: apiMethod,
          headers: parseHeaders(),
          auth: apiAuth,
        }
        return api.createApiTool({ ...base, api: apiCfg })
      }
      return api.createRagTool(base)
    },
    onSuccess: (tool) => {
      toast.success(`Tool "${tool.name}" created`)
      qc.invalidateQueries({ queryKey: ["tools"] })
      onOpenChange(false)
      resetForm()
    },
    onError: (e: ApiError) => toast.error(e.message),
  })

  function resetForm() {
    setName(""); setDescription(""); setMode("never"); setRisk("low"); setApprovalPrompt("")
    setMcpServerKey(""); setMcpToolName("")
    setApiBaseUrl(""); setApiMethod("GET"); setApiHeadersRaw(""); setApiAuth(emptyAuth())
    setTab("mcp")
  }

  return (
    <Dialog open={open} onOpenChange={(v) => { onOpenChange(v); if (!v) resetForm() }}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Create Tool</DialogTitle>
          <DialogDescription>Register a tool agents can use.</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label>Tool name</Label>
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Serpapi Search" />
            {toolKey && (
              <p className="text-xs text-muted-foreground">
                Key: <span className="font-mono">{toolKey}</span>
              </p>
            )}
          </div>

          <div className="space-y-1.5">
            <Label>Description</Label>
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What does this tool do?"
              rows={2}
            />
          </div>

          <Separator />

          <Tabs value={tab} onValueChange={(v) => setTab(v as ToolType)}>
            <TabsList className="w-full">
              <TabsTrigger value="mcp" className="flex-1">MCP</TabsTrigger>
              <TabsTrigger value="api" className="flex-1">API</TabsTrigger>
              <TabsTrigger value="rag" className="flex-1">RAG</TabsTrigger>
            </TabsList>

            <TabsContent value="mcp" className="space-y-4 mt-4">
              <div className="space-y-1.5">
                <Label>MCP Server</Label>
                <Select value={mcpServerKey} onValueChange={setMcpServerKey}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select server…" />
                  </SelectTrigger>
                  <SelectContent>
                    {(mcpServers ?? []).map((s) => (
                      <SelectItem key={s.server_key} value={s.server_key}>
                        {s.name}
                        <span className="ml-2 font-mono text-xs text-muted-foreground">
                          {s.server_key}
                        </span>
                      </SelectItem>
                    ))}
                    {!mcpServers?.length && (
                      <div className="px-2 py-1.5 text-xs text-muted-foreground">
                        No MCP servers registered
                      </div>
                    )}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label>Tool name on server</Label>
                <Input
                  value={mcpToolName}
                  onChange={(e) => setMcpToolName(e.target.value)}
                  placeholder="search"
                />
              </div>
            </TabsContent>

            <TabsContent value="api" className="space-y-4 mt-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="col-span-2 space-y-1.5">
                  <Label>Base URL</Label>
                  <Input
                    value={apiBaseUrl}
                    onChange={(e) => setApiBaseUrl(e.target.value)}
                    placeholder="https://api.example.com/v1"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Method</Label>
                  <Select value={apiMethod} onValueChange={(v) => setApiMethod(v as "GET" | "POST")}>
                    <SelectTrigger className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="GET">GET</SelectItem>
                      <SelectItem value="POST">POST</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="space-y-1.5">
                <Label>
                  Headers{" "}
                  <span className="text-xs text-muted-foreground font-normal">
                    (one per line, Key: Value)
                  </span>
                </Label>
                <Textarea
                  value={apiHeadersRaw}
                  onChange={(e) => setApiHeadersRaw(e.target.value)}
                  placeholder={"Content-Type: application/json\nX-Custom: value"}
                  rows={3}
                  className="font-mono text-xs"
                />
              </div>
              <Separator />
              <AuthBuilder value={apiAuth} onChange={setApiAuth} />
            </TabsContent>

            <TabsContent value="rag" className="mt-4">
              <p className="text-sm text-muted-foreground">
                RAG tools use the description as the retrieval query source. No additional config needed — the backend resolves the index from the tool key.
              </p>
            </TabsContent>
          </Tabs>

          <Separator />

          <ApprovalFields
            mode={mode}
            risk={risk}
            prompt={approvalPrompt}
            onChange={(p) => {
              if (p.mode !== undefined) setMode(p.mode)
              if (p.risk !== undefined) setRisk(p.risk)
              if (p.prompt !== undefined) setApprovalPrompt(p.prompt)
            }}
          />
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button disabled={!isValid || create.isPending} onClick={() => create.mutate()}>
            {create.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
            Create tool
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------
// Edit tool modal (PUT /tools/{key})
// ---------------------------------------------------------------------------

function EditToolModal({
  tool,
  open,
  onOpenChange,
}: {
  tool: ToolConfig
  open: boolean
  onOpenChange: (v: boolean) => void
}) {
  const qc = useQueryClient()
  const [name, setName] = useState(tool.name)
  const [description, setDescription] = useState(tool.description)
  const [mode, setMode] = useState<ApprovalMode>(tool.approval_mode)
  const [risk, setRisk] = useState<ApprovalRisk>(tool.approval_risk)
  const [approvalPrompt, setApprovalPrompt] = useState(tool.approval_prompt ?? "")

  const save = useMutation({
    mutationFn: () =>
      api.updateTool(tool.tool_key, {
        ...tool,
        name: name.trim(),
        description: description.trim(),
        approval_mode: mode,
        approval_risk: risk,
        approval_prompt: mode === "conditional" ? approvalPrompt.trim() || null : null,
      }),
    onSuccess: () => {
      toast.success("Tool updated")
      qc.invalidateQueries({ queryKey: ["tools"] })
      onOpenChange(false)
    },
    onError: (e: ApiError) => toast.error(e.message),
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Edit Tool</DialogTitle>
          <DialogDescription className="font-mono text-xs">{tool.tool_key}</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label>Name</Label>
            <Input value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="space-y-1.5">
            <Label>Description</Label>
            <Textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3} />
          </div>

          <Separator />

          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs capitalize">{tool.type}</Badge>
            <span className="text-xs text-muted-foreground">Type cannot be changed after creation.</span>
          </div>

          <ApprovalFields
            mode={mode}
            risk={risk}
            prompt={approvalPrompt}
            onChange={(p) => {
              if (p.mode !== undefined) setMode(p.mode)
              if (p.risk !== undefined) setRisk(p.risk)
              if (p.prompt !== undefined) setApprovalPrompt(p.prompt)
            }}
          />
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button disabled={save.isPending || !name.trim()} onClick={() => save.mutate()}>
            {save.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------
// ToolsPage
// ---------------------------------------------------------------------------

export function ToolsPage() {
  const qc = useQueryClient()
  const [createOpen, setCreateOpen] = useState(false)
  const [editTool, setEditTool] = useState<ToolConfig | null>(null)
  const [toggling, setToggling] = useState<string | null>(null)
  const [serverFilter, setServerFilter] = useState("all")

  const { data: tools, isLoading, isError } = useQuery({
    queryKey: ["tools"],
    queryFn: api.listTools,
  })

  const serverKeys = useMemo(() => {
    const keys = new Set((tools ?? []).map((t) => t.mcp?.server_key).filter(Boolean) as string[])
    return Array.from(keys).sort()
  }, [tools])

  const visibleTools = useMemo(() => {
    if (serverFilter === "all") return tools ?? []
    return (tools ?? []).filter((t) => t.mcp?.server_key === serverFilter)
  }, [tools, serverFilter])

  const toggle = useMutation({
    mutationFn: ({ key, enabled }: { key: string; enabled: boolean }) =>
      api.setToolEnabled(key, enabled),
    onMutate: ({ key }) => setToggling(key),
    onSettled: () => setToggling(null),
    onSuccess: (updated) => {
      qc.setQueryData<ToolConfig[]>(["tools"], (prev) =>
        prev?.map((t) => (t.tool_key === updated.tool_key ? updated : t)) ?? [],
      )
    },
    onError: (e: ApiError) => toast.error(e.message),
  })

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <PageHeader
        title="Tools"
        description="MCP, API, and RAG tools available to agents."
        action={
          <Button onClick={() => setCreateOpen(true)}>
            <Plus className="h-4 w-4" />
            Create tool
          </Button>
        }
      />

      {isLoading && (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )}

      {isError && (
        <p className="py-10 text-center text-sm text-destructive">Failed to load tools.</p>
      )}

      {!isLoading && !isError && tools?.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <Wrench className="mb-4 h-10 w-10 text-muted-foreground/40" />
          <h3 className="text-lg font-medium">No tools yet</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Create MCP, API, or RAG tools and assign them to agents.
          </p>
          <Button className="mt-4" onClick={() => setCreateOpen(true)}>
            <Plus className="h-4 w-4" />
            Create tool
          </Button>
        </div>
      )}

      {tools && tools.length > 0 && serverKeys.length > 1 && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground shrink-0">Server</span>
          <div className="relative">
            <select
              value={serverFilter}
              onChange={(e) => setServerFilter(e.target.value)}
              className="appearance-none rounded-md border border-border bg-background pl-2 pr-7 py-1 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            >
              <option value="all">All</option>
              {serverKeys.map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
            <svg className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      )}

      {tools && tools.length > 0 && (
        <div className="rounded-lg border border-border divide-y divide-border overflow-hidden">
          {visibleTools.map((t) => (
              <ToolCard
                key={t.tool_key}
                tool={t}
                toggling={toggling === t.tool_key}
                onToggle={(enabled) => toggle.mutate({ key: t.tool_key, enabled })}
                onEdit={() => setEditTool(t)}
              />
            ))}
        </div>
      )}

      <CreateToolModal open={createOpen} onOpenChange={setCreateOpen} />
      {editTool && (
        <EditToolModal
          key={editTool.tool_key}
          tool={editTool}
          open={!!editTool}
          onOpenChange={(v) => { if (!v) setEditTool(null) }}
        />
      )}
    </div>
  )
}
