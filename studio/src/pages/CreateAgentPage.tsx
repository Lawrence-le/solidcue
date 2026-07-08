import { useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { ArrowLeft, ArrowRight, Loader2 } from "lucide-react"
import { toast } from "sonner"
import { api, ApiError } from "@/lib/api"
import {
  emptyRole,
  normalizeKey,
  resolveBaseUrl,
  type RoleForm as RoleFormData,
} from "@/lib/agent-config"
import { RoleForm } from "@/components/RoleForm"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Switch } from "@/components/ui/switch"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"

const STEPS = ["Identity", "Provider roles", "Tools & review"]

export function CreateAgentPage() {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [step, setStep] = useState(0)

  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [keyTasks, setKeyTasks] = useState("")
  const [producesArtifacts, setProducesArtifacts] = useState(false)
  const [artifactDestination, setArtifactDestination] = useState("")
  const [identityTouched, setIdentityTouched] = useState(false)
  const [identitySubmitError, setIdentitySubmitError] = useState("")
  const [brain, setBrain] = useState<RoleFormData>(emptyRole("0.3"))
  const [lite, setLite] = useState<RoleFormData>(emptyRole("0.1"))
  const [reviewer, setReviewer] = useState<RoleFormData>(emptyRole("0.1"))
  const [writer, setWriter] = useState<RoleFormData>(emptyRole("0.7"))
  const [liteInherits, setLiteInherits] = useState(true)
  const [reviewerInherits, setReviewerInherits] = useState(true)
  const [writerInherits, setWriterInherits] = useState(true)
  const [selectedTools, setSelectedTools] = useState<string[]>([])
  const [toolQuery, setToolQuery] = useState("")

  const agentKey = useMemo(() => normalizeKey(name), [name])

  const tools = useQuery({ queryKey: ["tools"], queryFn: api.listTools })
  const enabledTools = (tools.data ?? []).filter((t) => t.enabled)
  const filteredTools = enabledTools.filter((tool) => {
    const query = toolQuery.trim().toLowerCase()
    if (!query) return true
    return (
      tool.tool_key.toLowerCase().includes(query) ||
      tool.name.toLowerCase().includes(query) ||
      tool.type.toLowerCase().includes(query) ||
      tool.description.toLowerCase().includes(query) ||
      getToolServerLabel(tool).toLowerCase().includes(query)
    )
  })

  const resolvedLite = liteInherits ? inheritRole(brain, lite.temperature || "0.1") : lite
  const resolvedReviewer = reviewerInherits
    ? inheritRole(brain, reviewer.temperature || "0.1")
    : reviewer
  const resolvedWriter = writerInherits ? inheritRole(brain, writer.temperature || "0.7") : writer

  const handleLiteInheritsChange = (checked: boolean) => {
    setLiteInherits(checked)
    if (!checked) setLite((current) => seedCustomRole(current, brain, "0.1"))
  }

  const handleReviewerInheritsChange = (checked: boolean) => {
    setReviewerInherits(checked)
    if (!checked) setReviewer((current) => seedCustomRole(current, brain, "0.1"))
  }

  const handleWriterInheritsChange = (checked: boolean) => {
    setWriterInherits(checked)
    if (!checked) setWriter((current) => seedCustomRole(current, brain, "0.7"))
  }

  const create = useMutation({
    mutationFn: () => {
      const payload = {
        name: name.trim(),
        agent_key: agentKey,
        description: description.trim(),
        decision_provider_type: brain.provider_type,
        decision_base_url: resolveBaseUrl(brain),
        decision_api_key: brain.api_key,
        decision_model: brain.model.trim(),
        decision_temperature: Number(brain.temperature),
        lite_provider_type: resolvedLite.provider_type,
        lite_base_url: resolveBaseUrl(resolvedLite),
        lite_api_key: resolvedLite.api_key,
        lite_model: resolvedLite.model.trim(),
        lite_temperature: Number(resolvedLite.temperature),
        reviewer_provider_type: resolvedReviewer.provider_type,
        reviewer_base_url: resolveBaseUrl(resolvedReviewer),
        reviewer_api_key: resolvedReviewer.api_key,
        reviewer_model: resolvedReviewer.model.trim(),
        reviewer_temperature: Number(resolvedReviewer.temperature),
        ...(writerInherits
          ? {}
          : {
              writer_provider_type: resolvedWriter.provider_type,
              writer_base_url: resolveBaseUrl(resolvedWriter),
              writer_api_key: resolvedWriter.api_key,
              writer_model: resolvedWriter.model.trim(),
              writer_temperature: Number(resolvedWriter.temperature),
            }),
        selected_tools: selectedTools,
        key_tasks: keyTasks
          .split("\n")
          .map((t) => t.trim())
          .filter(Boolean),
        produces_artifacts: producesArtifacts,
        ...(producesArtifacts && artifactDestination.trim()
          ? { artifact_destination: artifactDestination.trim() }
          : {}),
      }
      return api.createAgent(payload)
    },
    onSuccess: (agent) => {
      toast.success(`Created agent: ${agent.name}`)
      qc.invalidateQueries({ queryKey: ["agents"] })
      navigate("/agents")
    },
    onError: (e: ApiError) => {
      if (e.status === 409) {
        setStep(0)
        setIdentitySubmitError("Agent key already exists")
        return
      }
      toast.error(e.message)
    },
  })

  // Step validation
  const trimmedName = name.trim()
  const nameError = trimmedName.length === 0 ? "Name is required" : ""
  const keyError =
    !nameError && agentKey.length === 0
      ? "Name does not produce a valid key. Use letters or numbers."
      : ""
  const step1FieldError = nameError || keyError
  const step1Error = identitySubmitError || (identityTouched ? step1FieldError : "")
  const step1Valid = !step1FieldError
  const roleValid = (r: RoleFormData) =>
    r.model.trim().length > 0 && r.api_key.length > 0 && r.temperature.trim() !== "" &&
    (PROVIDER_NEEDS_BASE(r) ? r.base_url.trim().length > 0 : true)
  const step2Valid =
    roleValid(brain) &&
    (liteInherits || roleValid(lite)) &&
    (reviewerInherits || roleValid(reviewer)) &&
    (writerInherits || roleValid(writer))

  const canNext = step === 0 ? step1Valid : step === 1 ? step2Valid : true

  return (
    <div className="mx-auto max-w-3xl">
      <button
        onClick={() => navigate("/agents")}
        className="mb-4 flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to agents
      </button>

      <h1 className="text-2xl font-semibold tracking-tight">Create Agent</h1>

      {/* Stepper */}
      <div className="my-6 overflow-x-auto px-2 py-2">
        <div className="flex min-w-max items-start gap-4">
        {STEPS.map((label, i) => (
          <div key={label} className="flex items-start gap-4 shrink-0">
            <div className="flex flex-col items-center gap-2">
              <div
                className={cn(
                  "flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-medium tabular-nums",
                  i < step && "bg-primary text-primary-foreground",
                  i === step && "bg-primary text-primary-foreground ring-2 ring-ring ring-offset-2 ring-offset-background",
                  i > step && "bg-muted text-muted-foreground",
                )}
              >
                {i + 1}
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <div className="flex h-6 items-center">
                {i < STEPS.length - 1 && <Separator className="w-10 shrink-0" />}
              </div>
              <span
                className={cn(
                  "whitespace-nowrap text-sm",
                  i === step ? "font-medium" : "text-muted-foreground",
                )}
              >
                {label}
              </span>
            </div>
          </div>
        ))}
        </div>
      </div>

      <Card className="p-6">
        {step === 0 && (
          <div className="space-y-4">
            <div className="space-y-1.5">
              <Label>Agent name</Label>
              <Input
                value={name}
                onChange={(e) => {
                  if (!identityTouched) setIdentityTouched(true)
                  setName(e.target.value)
                  if (identitySubmitError) setIdentitySubmitError("")
                }}
                placeholder="Resume Builder"
                aria-invalid={!!step1Error}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Agent key</Label>
              <Input value={agentKey} readOnly aria-invalid={!!step1Error} className="font-mono" />
              <p className="text-xs text-muted-foreground">
                Generated from name. Uses lowercase letters, numbers, and underscores.
              </p>
              {step1Error && <p className="text-xs text-destructive">{step1Error}</p>}
            </div>
            <div className="space-y-1.5">
              <Label>Description</Label>
              <Textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="What does this agent do?"
                rows={3}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Key tasks</Label>
              <Textarea
                value={keyTasks}
                onChange={(e) => setKeyTasks(e.target.value)}
                placeholder={"One task per line, e.g.\nScrape the job posting\nSave a formatted summary"}
                rows={3}
              />
              <p className="text-xs text-muted-foreground">
                The agent&apos;s main tasks — one per line. Grounds the generated SKILL.md.
              </p>
            </div>
            <div className="flex items-center justify-between gap-3">
              <div>
                <Label htmlFor="produces-artifacts">Produces saved artifacts</Label>
                <p className="text-xs text-muted-foreground">
                  Does this agent output files or documents it saves somewhere?
                </p>
              </div>
              <Switch
                id="produces-artifacts"
                checked={producesArtifacts}
                onCheckedChange={setProducesArtifacts}
              />
            </div>
            {producesArtifacts && (
              <div className="space-y-1.5">
                <Label>Artifact destination</Label>
                <Input
                  value={artifactDestination}
                  onChange={(e) => setArtifactDestination(e.target.value)}
                  placeholder="e.g. drive://Recruiting/JDs/{date}-jd.docx"
                  className="font-mono"
                />
                <p className="text-xs text-muted-foreground">
                  Exact path and/or filename where outputs are saved. Written verbatim into
                  SKILL.md and TOOLS.md.
                </p>
              </div>
            )}
          </div>
        )}

        {step === 1 && (
          <div className="space-y-8">
            <RoleSection title="Brain (decision maker)" desc="Main reasoning model.">
              <RoleForm value={brain} onChange={setBrain} />
            </RoleSection>
            <RoleSection
              title="Lite"
              desc="Fast/cheap model for light tasks."
              right={
                <InheritToggle
                  id="lite-inherit"
                  checked={liteInherits}
                  onCheckedChange={handleLiteInheritsChange}
                  label="Use Brain settings"
                />
              }
            >
              {liteInherits ? (
                <InheritedRoleSummary role={resolvedLite} temperatureLabel="0.1 default" />
              ) : (
                <RoleForm value={lite} onChange={setLite} />
              )}
            </RoleSection>
            <RoleSection
              title="Reviewer"
              desc="Evaluates drafts and quality."
              right={
                <InheritToggle
                  id="reviewer-inherit"
                  checked={reviewerInherits}
                  onCheckedChange={handleReviewerInheritsChange}
                  label="Use Brain settings"
                />
              }
            >
              {reviewerInherits ? (
                <InheritedRoleSummary role={resolvedReviewer} temperatureLabel="0.1 default" />
              ) : (
                <RoleForm value={reviewer} onChange={setReviewer} />
              )}
            </RoleSection>
            <RoleSection
              title="Writer (synthesis)"
              desc="High-quality output generation."
              right={
                <InheritToggle
                  id="writer-inherit"
                  checked={writerInherits}
                  onCheckedChange={handleWriterInheritsChange}
                  label="Use Brain settings"
                />
              }
            >
              {!writerInherits && <RoleForm value={writer} onChange={setWriter} />}
              {writerInherits && (
                <InheritedRoleSummary role={resolvedWriter} temperatureLabel="0.7 default" />
              )}
            </RoleSection>
          </div>
        )}

        {step === 2 && (
          <div className="space-y-6">
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-3">
                <Label>Tools</Label>
                <span className="text-xs text-muted-foreground">
                  {selectedTools.length} selected
                </span>
              </div>
              {tools.isLoading && <p className="text-sm text-muted-foreground">Loading tools…</p>}
              {tools.isSuccess && enabledTools.length === 0 && (
                <p className="text-sm text-muted-foreground">
                  No tools registered. The agent will be created without tools.
                </p>
              )}
              {enabledTools.length > 0 && (
                <>
                  <Input
                    value={toolQuery}
                    onChange={(e) => setToolQuery(e.target.value)}
                    placeholder="Search tools by name, key, type, or description"
                  />
                  {selectedTools.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {selectedTools.map((toolKey) => (
                        <button
                          key={toolKey}
                          type="button"
                          onClick={() =>
                            setSelectedTools((prev) => prev.filter((k) => k !== toolKey))
                          }
                          className="rounded-full border border-primary bg-primary/10 px-3 py-1 text-sm text-primary transition-colors"
                        >
                          {toolKey}
                        </button>
                      ))}
                    </div>
                  )}
                  <div className="space-y-2">
                    {filteredTools.length === 0 ? (
                      <p className="text-sm text-muted-foreground">No tools match your search.</p>
                    ) : (
                      filteredTools.map((t) => {
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
                            className={cn(
                              "w-full rounded-lg border px-4 py-3 text-left transition-colors",
                              on
                                ? "border-primary bg-primary/5"
                                : "border-border hover:bg-accent/40",
                            )}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0 space-y-1">
                                <div className="flex items-center gap-2">
                                  <span className="text-sm font-medium">{t.name}</span>
                                  <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] uppercase tracking-wide text-muted-foreground">
                                    {t.type}
                                  </span>
                                </div>
                                <p className="font-mono text-xs text-muted-foreground">{t.tool_key}</p>
                                <p className="text-xs text-muted-foreground">
                                  Server: <span className="font-mono">{getToolServerLabel(t)}</span>
                                </p>
                                <p className="text-sm text-muted-foreground">
                                  {t.description || "No description available."}
                                </p>
                              </div>
                              <span
                                className={cn(
                                  "shrink-0 rounded-full px-2 py-1 text-xs",
                                  on
                                    ? "bg-primary text-primary-foreground"
                                    : "bg-muted text-muted-foreground",
                                )}
                              >
                                {on ? "Selected" : "Select"}
                              </span>
                            </div>
                          </button>
                        )
                      })
                    )}
                  </div>
                </>
              )}
            </div>

            <Separator />

            <div className="space-y-2 text-sm">
              <h3 className="font-medium">Review</h3>
              <ReviewRow label="Name" value={name} />
              <ReviewRow label="Key" value={agentKey} mono />
              <ReviewRow label="Brain" value={formatRoleReview(brain)} />
              <ReviewRow label="Lite" value={formatRoleReview(resolvedLite, liteInherits)} />
              <ReviewRow label="Reviewer" value={formatRoleReview(resolvedReviewer, reviewerInherits)} />
              <ReviewRow label="Writer" value={formatRoleReview(resolvedWriter, writerInherits)} />
              <ReviewRow label="Tools" value={selectedTools.length ? selectedTools.join(", ") : "none"} />
              <ReviewRow
                label="Tasks"
                value={
                  keyTasks.split("\n").map((t) => t.trim()).filter(Boolean).join(", ") || "none"
                }
              />
              <ReviewRow
                label="Artifacts"
                value={producesArtifacts ? artifactDestination.trim() || "yes (no destination)" : "no"}
              />
            </div>
          </div>
        )}
      </Card>

      <div className="mt-6 flex justify-between">
        <Button
          variant="outline"
          onClick={() => (step === 0 ? navigate("/agents") : setStep((s) => s - 1))}
        >
          {step === 0 ? "Cancel" : "Back"}
        </Button>
        {step < STEPS.length - 1 ? (
          <Button disabled={!canNext} onClick={() => setStep((s) => s + 1)}>
            Next
            <ArrowRight className="h-4 w-4" />
          </Button>
        ) : (
          <Button disabled={create.isPending} onClick={() => create.mutate()}>
            {create.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
            Create agent
          </Button>
        )}
      </div>
    </div>
  )
}

function PROVIDER_NEEDS_BASE(r: RoleFormData): boolean {
  // openai_compatible requires a base url
  return r.provider_type === "openai_compatible"
}

function inheritRole(brain: RoleFormData, temperature: string): RoleFormData {
  return {
    ...brain,
    temperature,
  }
}

function seedCustomRole(current: RoleFormData, brain: RoleFormData, temperature: string): RoleFormData {
  return {
    provider_type: current.provider_type || brain.provider_type,
    base_url: current.base_url || brain.base_url,
    api_key: current.api_key || brain.api_key,
    model: current.model || brain.model,
    temperature: current.temperature || temperature,
  }
}

function InheritToggle({
  id,
  checked,
  onCheckedChange,
  label,
}: {
  id: string
  checked: boolean
  onCheckedChange: (checked: boolean) => void
  label: string
}) {
  return (
    <div className="flex items-center gap-2">
      <Label htmlFor={id} className="text-xs text-muted-foreground">
        {label}
      </Label>
      <Switch id={id} checked={checked} onCheckedChange={onCheckedChange} />
    </div>
  )
}

function InheritedRoleSummary({
  role,
  temperatureLabel,
}: {
  role: RoleFormData
  temperatureLabel: string
}) {
  return (
    <div className="rounded-lg border border-dashed border-border bg-muted/30 px-4 py-3 text-sm">
      <p className="text-muted-foreground">This role will use the Brain provider settings.</p>
      <div className="mt-2 grid gap-1 text-xs text-muted-foreground">
        <div>
          <span className="font-medium text-foreground">Provider:</span> {role.provider_type}
        </div>
        <div>
          <span className="font-medium text-foreground">Model:</span>{" "}
          <span className="font-mono">{role.model || "Set Brain model first"}</span>
        </div>
        <div>
          <span className="font-medium text-foreground">Temperature:</span>{" "}
          <span className="font-mono">{role.temperature}</span>
          <span className="ml-1">({temperatureLabel})</span>
        </div>
      </div>
    </div>
  )
}

function RoleSection({
  title,
  desc,
  right,
  children,
}: {
  title: string
  desc: string
  right?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-medium">{title}</h3>
          <p className="text-xs text-muted-foreground">{desc}</p>
        </div>
        {right}
      </div>
      {children}
    </div>
  )
}

function ReviewRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex gap-3">
      <span className="w-20 shrink-0 text-muted-foreground">{label}</span>
      <span className={cn("truncate", mono && "font-mono")}>{value}</span>
    </div>
  )
}

function formatRoleReview(role: RoleFormData, inherits = false): string {
  const prefix = inherits ? "inherits Brain · " : ""
  return `${prefix}${role.provider_type} · ${role.model} · ${role.temperature}`
}

function getToolServerLabel(tool: { type: string; mcp: { server_key: string } | null }): string {
  if (tool.type === "mcp" && tool.mcp?.server_key) return tool.mcp.server_key
  if (tool.type === "api") return "api"
  if (tool.type === "rag") return "rag"
  return "direct"
}
