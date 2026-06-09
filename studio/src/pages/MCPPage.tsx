import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Loader2, Plus, Server } from "lucide-react"
import { toast } from "sonner"
import { api, ApiError } from "@/lib/api"
import type { MCPAuthConfig, MCPServerConfig } from "@/lib/types"
import { normalizeKey } from "@/lib/agent-config"
import { AuthBuilder, authValid, emptyAuth } from "@/components/AuthBuilder"
import { MCPServerCard } from "@/components/MCPServerCard"
import { PageHeader } from "@/components/PageHeader"
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
import { Textarea } from "@/components/ui/textarea"
import { Separator } from "@/components/ui/separator"

function AddMcpServerModal({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
}) {
  const qc = useQueryClient()
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [url, setUrl] = useState("")
  const [auth, setAuth] = useState<MCPAuthConfig>(emptyAuth())

  const serverKey = normalizeKey(name)

  const isValid =
    name.trim().length > 0 &&
    serverKey.length > 0 &&
    url.trim().length > 0 &&
    authValid(auth)

  const create = useMutation({
    mutationFn: () => {
      const payload: MCPServerConfig = {
        server_key: serverKey,
        name: name.trim(),
        description: description.trim(),
        transport: "streamable_http",
        url: url.trim(),
        auth,
        enabled: true,
      }
      return api.createMcpServer(payload)
    },
    onSuccess: (server) => {
      toast.success(`Added MCP server: ${server.name}`)
      qc.invalidateQueries({ queryKey: ["mcp-servers"] })
      onOpenChange(false)
      // reset
      setName("")
      setDescription("")
      setUrl("")
      setAuth(emptyAuth())
    },
    onError: (e: ApiError) => toast.error(e.message),
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Add MCP Server</DialogTitle>
          <DialogDescription>Register a Streamable HTTP MCP server.</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label>Server name</Label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My MCP Server"
            />
            {serverKey && (
              <p className="text-xs text-muted-foreground">
                Key: <span className="font-mono">{serverKey}</span>
              </p>
            )}
          </div>

          <div className="space-y-1.5">
            <Label>Description</Label>
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What tools does this server expose?"
              rows={2}
            />
          </div>

          <div className="space-y-1.5">
            <Label>URL</Label>
            <Input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://mcp.example.com/sse"
            />
          </div>

          <Separator />

          <AuthBuilder value={auth} onChange={setAuth} />
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button disabled={!isValid || create.isPending} onClick={() => create.mutate()}>
            {create.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
            Add server
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export function MCPPage() {
  const [modalOpen, setModalOpen] = useState(false)
  const { data: servers, isLoading, isError } = useQuery({
    queryKey: ["mcp-servers"],
    queryFn: api.listMcpServers,
  })

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <PageHeader
        title="MCP Servers"
        description="Streamable HTTP servers that expose tools to your agents."
        action={
          <Button onClick={() => setModalOpen(true)}>
            <Plus className="h-4 w-4" />
            Add server
          </Button>
        }
      />

      {isLoading && (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )}

      {isError && (
        <p className="py-10 text-center text-sm text-destructive">Failed to load MCP servers.</p>
      )}

      {!isLoading && !isError && servers?.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <Server className="mb-4 h-10 w-10 text-muted-foreground/40" />
          <h3 className="text-lg font-medium">No MCP servers</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Add a Streamable HTTP MCP server to expose its tools to agents.
          </p>
          <Button className="mt-4" onClick={() => setModalOpen(true)}>
            <Plus className="h-4 w-4" />
            Add server
          </Button>
        </div>
      )}

      {servers && servers.length > 0 && (
        <div className="rounded-lg border border-border divide-y divide-border overflow-hidden">
          {servers.map((s) => (
            <MCPServerCard key={s.server_key} server={s} />
          ))}
        </div>
      )}

      <AddMcpServerModal open={modalOpen} onOpenChange={setModalOpen} />
    </div>
  )
}
