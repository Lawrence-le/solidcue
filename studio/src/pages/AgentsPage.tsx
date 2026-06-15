import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Loader2, Plus } from "lucide-react"
import { toast } from "sonner"
import { api, ApiError } from "@/lib/api"
import type { AgentConfig } from "@/lib/types"
import {
  emptyRole,
  resolveBaseUrl,
  type RoleForm as RoleFormData,
  type ProviderType,
} from "@/lib/agent-config"
import { RoleForm } from "@/components/RoleForm"
import { PageHeader } from "@/components/PageHeader"
import { AgentCard } from "@/components/AgentCard"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Switch } from "@/components/ui/switch"
import { Separator } from "@/components/ui/separator"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"

// ---------------------------------------------------------------------------
// Discover dialog — read-only view of all providers + tools
// ---------------------------------------------------------------------------

const ROLE_LABELS = [
  { label: "Brain", key: "provider" },
  { label: "Lite", key: "lite_provider" },
  { label: "Reviewer", key: "reviewer_provider" },
  { label: "Writer", key: "writer_provider" },
] as const

function formatTemperature(temperature: number | null): string {
  return temperature == null ? "n/a" : String(temperature)
}

function DiscoverAgentDialog({
  agent,
  open,
  onOpenChange,
}: {
  agent: AgentConfig
  open: boolean
  onOpenChange: (v: boolean) => void
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex h-[min(80vh,40rem)] max-w-lg flex-col overflow-hidden">
        <DialogHeader>
          <DialogTitle>{agent.name}</DialogTitle>
          <DialogDescription className="font-mono text-xs">{agent.agent_key}</DialogDescription>
        </DialogHeader>

        <ScrollArea className="min-h-0 flex-1 pr-4">
          <div className="space-y-5">
            {/* Providers */}
            <div>
              <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">Providers</p>
              <div className="rounded-lg border border-border divide-y divide-border overflow-hidden">
                {ROLE_LABELS.map(({ label, key }) => {
                  const p = agent[key] ?? agent.provider
                  const inheritsBrain = key !== "provider" && agent[key] == null
                  return (
                    <div key={label} className="flex items-start gap-4 px-3 py-2.5">
                      <span className="w-20 shrink-0 text-xs text-muted-foreground">{label}</span>
                      <div className="min-w-0 flex-1 space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-medium">{p.type}</span>
                          {inheritsBrain && (
                            <span className="text-xs text-muted-foreground italic">inherits Brain</span>
                          )}
                        </div>
                        <div className="grid gap-1 text-xs text-muted-foreground">
                          <div>
                            <span className="font-medium text-foreground">Model:</span>{" "}
                            <span className="font-mono break-all">{p.model}</span>
                          </div>
                          <div>
                            <span className="font-medium text-foreground">Temperature:</span>{" "}
                            <span className="font-mono">{formatTemperature(p.temperature)}</span>
                          </div>
                          {p.base_url && (
                            <div>
                              <span className="font-medium text-foreground">Base URL:</span>{" "}
                              <span className="font-mono break-all">{p.base_url}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Tools */}
            <div>
              <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Tools ({agent.tools.length})
              </p>
              {agent.tools.length === 0 ? (
                <p className="text-xs text-muted-foreground italic">No tools assigned</p>
              ) : (
                <div className="rounded-lg border border-border divide-y divide-border overflow-hidden">
                  {agent.tools.map((t) => (
                    <div key={t} className="px-3 py-2">
                      <span className="font-mono text-xs">{t}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------
// Edit agent dialog
// ---------------------------------------------------------------------------

function providerToRoleForm(p: AgentConfig["provider"]): RoleFormData {
  return {
    provider_type: (p.type as ProviderType) ?? "openai_compatible",
    base_url: p.base_url ?? "",
    api_key: "",
    model: p.model,
    temperature: String(p.temperature ?? ""),
  }
}

function RoleSection({
  title,
  desc,
  right,
  children,
}: {
  title: string
  desc?: string
  right?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium">{title}</p>
          {desc && <p className="text-xs text-muted-foreground">{desc}</p>}
        </div>
        {right}
      </div>
      {children}
    </div>
  )
}

function PROVIDER_NEEDS_BASE(r: RoleFormData): boolean {
  return r.provider_type === "openai_compatible"
}

function EditAgentDialog({
  agent,
  open,
  onOpenChange,
}: {
  agent: AgentConfig
  open: boolean
  onOpenChange: (v: boolean) => void
}) {
  const qc = useQueryClient()

  const [name, setName] = useState(agent.name)
  const [description, setDescription] = useState(agent.description)
  const [brain, setBrain] = useState<RoleFormData>(() => providerToRoleForm(agent.provider))
  const [lite, setLite] = useState<RoleFormData>(() =>
    agent.lite_provider ? providerToRoleForm(agent.lite_provider) : emptyRole("0.1"),
  )
  const [reviewer, setReviewer] = useState<RoleFormData>(() =>
    agent.reviewer_provider ? providerToRoleForm(agent.reviewer_provider) : emptyRole("0.1"),
  )
  const [writerInherits, setWriterInherits] = useState(!agent.writer_provider)
  const [writer, setWriter] = useState<RoleFormData>(() =>
    agent.writer_provider ? providerToRoleForm(agent.writer_provider) : emptyRole("0.7"),
  )

  const tools = useQuery({ queryKey: ["tools"], queryFn: api.listTools, enabled: open })
  const enabledTools = (tools.data ?? []).filter((t) => t.enabled)
  const [selectedTools, setSelectedTools] = useState<string[]>(agent.tools)

  const roleValid = (r: RoleFormData) =>
    r.model.trim().length > 0 &&
    r.temperature.trim() !== "" &&
    (PROVIDER_NEEDS_BASE(r) ? r.base_url.trim().length > 0 : true)

  const isValid =
    name.trim().length > 0 &&
    roleValid(brain) &&
    roleValid(lite) &&
    roleValid(reviewer) &&
    (writerInherits || roleValid(writer))

  const save = useMutation({
    mutationFn: () => {
      const payload = {
        name: name.trim(),
        description: description.trim(),
        decision_provider_type: brain.provider_type,
        decision_base_url: resolveBaseUrl(brain),
        decision_api_key: brain.api_key || null,
        decision_model: brain.model.trim(),
        decision_temperature: Number(brain.temperature),
        lite_provider_type: lite.provider_type,
        lite_base_url: resolveBaseUrl(lite),
        lite_api_key: lite.api_key || null,
        lite_model: lite.model.trim(),
        lite_temperature: Number(lite.temperature),
        reviewer_provider_type: reviewer.provider_type,
        reviewer_base_url: resolveBaseUrl(reviewer),
        reviewer_api_key: reviewer.api_key || null,
        reviewer_model: reviewer.model.trim(),
        reviewer_temperature: Number(reviewer.temperature),
        ...(writerInherits
          ? {}
          : {
              writer_provider_type: writer.provider_type,
              writer_base_url: resolveBaseUrl(writer),
              writer_api_key: writer.api_key || null,
              writer_model: writer.model.trim(),
              writer_temperature: Number(writer.temperature),
            }),
        selected_tools: selectedTools,
      }
      return api.updateAgent(agent.agent_key, payload)
    },
    onSuccess: () => {
      toast.success("Agent updated")
      qc.invalidateQueries({ queryKey: ["agents"] })
      onOpenChange(false)
    },
    onError: (e: ApiError) => toast.error(e.message),
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl overflow-hidden">
        <DialogHeader>
          <DialogTitle>Edit Agent</DialogTitle>
          <DialogDescription className="font-mono text-xs">{agent.agent_key}</DialogDescription>
        </DialogHeader>

        <ScrollArea className="max-h-[70vh] pr-1">
          <div className="space-y-6 pr-3">
            {/* Identity */}
            <div className="space-y-4">
              <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Identity</p>
              <div className="space-y-1.5">
                <Label>Name</Label>
                <Input value={name} onChange={(e) => setName(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label>Description</Label>
                <Textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={2} />
              </div>
            </div>

            <Separator />

            {/* Provider roles */}
            <div className="space-y-6">
              <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Providers — leave API key blank to keep existing
              </p>
              <RoleSection title="Brain" desc="Main reasoning model.">
                <RoleForm value={brain} onChange={setBrain} />
              </RoleSection>
              <RoleSection title="Lite" desc="Fast/cheap model for light tasks.">
                <RoleForm value={lite} onChange={setLite} />
              </RoleSection>
              <RoleSection title="Reviewer" desc="Evaluates drafts and quality.">
                <RoleForm value={reviewer} onChange={setReviewer} />
              </RoleSection>
              <RoleSection
                title="Writer"
                desc="High-quality output generation."
                right={
                  <div className="flex items-center gap-2">
                    <Label className="text-xs text-muted-foreground">Inherit from Brain</Label>
                    <Switch checked={writerInherits} onCheckedChange={setWriterInherits} />
                  </div>
                }
              >
                {!writerInherits && <RoleForm value={writer} onChange={setWriter} />}
                {writerInherits && (
                  <p className="text-sm text-muted-foreground">Writer will reuse the Brain provider.</p>
                )}
              </RoleSection>
            </div>

            <Separator />

            {/* Tools */}
            <div className="space-y-2">
              <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Tools</p>
              {tools.isLoading && <p className="text-sm text-muted-foreground">Loading tools…</p>}
              {tools.isSuccess && enabledTools.length === 0 && (
                <p className="text-sm text-muted-foreground">No tools registered.</p>
              )}
              <div className="flex flex-wrap gap-2">
                {enabledTools.map((t) => {
                  const on = selectedTools.includes(t.tool_key)
                  return (
                    <button
                      key={t.tool_key}
                      type="button"
                      onClick={() =>
                        setSelectedTools((prev) =>
                          on ? prev.filter((k) => k !== t.tool_key) : [...prev, t.tool_key],
                        )
                      }
                      className={`rounded-full border px-3 py-1 text-sm transition-colors ${
                        on
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {t.tool_key}
                      <span className="ml-1.5 text-xs opacity-60">[{t.type}]</span>
                    </button>
                  )
                })}
              </div>
            </div>
          </div>
        </ScrollArea>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button disabled={!isValid || save.isPending} onClick={() => save.mutate()}>
            {save.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
            Save changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------
// AgentsPage
// ---------------------------------------------------------------------------

export function AgentsPage() {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [pendingDelete, setPendingDelete] = useState<AgentConfig | null>(null)
  const [editAgent, setEditAgent] = useState<AgentConfig | null>(null)
  const [discoverAgent, setDiscoverAgent] = useState<AgentConfig | null>(null)

  const agents = useQuery({ queryKey: ["agents"], queryFn: api.listAgents })

  const del = useMutation({
    mutationFn: (key: string) => api.deleteAgent(key),
    onSuccess: () => {
      toast.success("Agent deleted")
      setPendingDelete(null)
      qc.invalidateQueries({ queryKey: ["agents"] })
    },
    onError: (e: ApiError) => toast.error(e.message),
  })

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <PageHeader
        title="Agents"
        description="Config-driven agents. Run, create, or remove."
        action={
          <Button onClick={() => navigate("/agents/new")}>
            <Plus className="h-4 w-4" />
            New
          </Button>
        }
      />

      {agents.isLoading && (
        <div className="space-y-2">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-14 w-full rounded-lg" />
          ))}
        </div>
      )}

      {agents.isError && (
        <p className="text-sm text-destructive">
          Failed to load agents: {(agents.error as ApiError).message}
        </p>
      )}

      {agents.isSuccess && agents.data.length === 0 && (
        <div className="rounded-lg border border-dashed border-border p-12 text-center">
          <p className="text-sm text-muted-foreground">No agents yet.</p>
          <Button className="mt-4" onClick={() => navigate("/agents/new")}>
            <Plus className="h-4 w-4" />
            Create your first agent
          </Button>
        </div>
      )}

      {agents.isSuccess && agents.data.length > 0 && (
        <div className="rounded-lg border border-border divide-y divide-border overflow-hidden">
          {agents.data.map((agent) => (
            <AgentCard
              key={agent.agent_id}
              agent={agent}
              onRun={(a) => navigate(`/sessions?agent=${a.agent_key}`)}
              onEdit={setEditAgent}
              onDiscover={setDiscoverAgent}
              onDelete={setPendingDelete}
            />
          ))}
        </div>
      )}

      {/* Discover dialog */}
      {discoverAgent && (
        <DiscoverAgentDialog
          agent={discoverAgent}
          open={!!discoverAgent}
          onOpenChange={(v) => { if (!v) setDiscoverAgent(null) }}
        />
      )}

      {/* Edit dialog */}
      {editAgent && (
        <EditAgentDialog
          key={editAgent.agent_key}
          agent={editAgent}
          open={!!editAgent}
          onOpenChange={(v) => { if (!v) setEditAgent(null) }}
        />
      )}

      {/* Delete confirmation */}
      <Dialog open={!!pendingDelete} onOpenChange={(o) => !o && setPendingDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete agent?</DialogTitle>
            <DialogDescription>
              This permanently deletes <span className="font-mono">{pendingDelete?.agent_key}</span>{" "}
              and its config files. This cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              disabled={del.isPending}
              onClick={() => pendingDelete && del.mutate(pendingDelete.agent_key)}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
