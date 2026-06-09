import { useState } from "react"
import { Globe, Loader2, CheckCircle2, Search } from "lucide-react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { toast } from "sonner"
import type { DiscoveredTool, MCPServerConfig } from "@/lib/types"
import { api, ApiError } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

const AUTH_LABELS: Record<string, string> = {
  none: "No auth",
  api_key: "API key",
  bearer: "Bearer",
  oauth: "OAuth",
}

function DiscoveredToolRow({
  tool,
  serverKey,
  isRegistered,
  onCreated,
}: {
  tool: DiscoveredTool
  serverKey: string
  isRegistered: boolean
  onCreated: () => void
}) {
  const name = tool.name ?? tool.title ?? "unknown"

  const create = useMutation({
    mutationFn: () =>
      api.createMcpTool({
        name,
        tool_key: name.toLowerCase().replace(/[^a-z0-9]+/g, "_"),
        description: tool.description ?? "",
        mcp: { server_key: serverKey, tool_name: name, input_schema: tool.input_schema ?? null },
      }),
    onSuccess: () => {
      toast.success(`Tool "${name}" created`)
      onCreated()
    },
    onError: (e: ApiError) => toast.error(e.message),
  })

  return (
    <div className="flex w-full items-center gap-4 py-2.5">
      <Tooltip>
        <TooltipTrigger asChild>
          <p className="min-w-0 flex-1 truncate font-mono text-sm font-medium cursor-default">
            {name}
          </p>
        </TooltipTrigger>
        {tool.description && (
          <TooltipContent side="top" className="max-w-xs text-xs">
            {tool.description}
          </TooltipContent>
        )}
      </Tooltip>
      {isRegistered ? (
        <div className="flex shrink-0 items-center gap-1 text-xs text-green-500">
          <CheckCircle2 className="h-3.5 w-3.5" />
          Registered
        </div>
      ) : (
        <Button
          size="sm"
          variant="outline"
          className="shrink-0"
          disabled={create.isPending}
          onClick={() => create.mutate()}
        >
          {create.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "+ Register"}
        </Button>
      )}
    </div>
  )
}

function DiscoverDialog({
  server,
  open,
  onOpenChange,
}: {
  server: MCPServerConfig
  open: boolean
  onOpenChange: (v: boolean) => void
}) {
  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["mcp-discovered", server.server_key],
    queryFn: () => api.discoveredTools(server.server_key),
    enabled: open,
    staleTime: 0,
  })

  const { data: existingTools } = useQuery({
    queryKey: ["tools"],
    queryFn: api.listTools,
    enabled: open,
  })

  const registeredToolNames = new Set(
    (existingTools ?? [])
      .filter((t) => t.mcp?.server_key === server.server_key)
      .map((t) => t.mcp!.tool_name),
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg overflow-hidden">
        <DialogHeader>
          <DialogTitle>Discovered tools — {server.name}</DialogTitle>
          <DialogDescription className="font-mono text-xs">{server.url}</DialogDescription>
        </DialogHeader>

        {isLoading && (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        )}

        {isError && (
          <div className="space-y-2 py-4 text-center">
            <p className="text-sm text-destructive">
              {(error as ApiError)?.message ?? "Failed to discover tools"}
            </p>
            <Button variant="outline" size="sm" onClick={() => refetch()}>
              Retry
            </Button>
          </div>
        )}

        {data && data.length === 0 && (
          <p className="py-6 text-center text-sm text-muted-foreground">
            No tools discovered from this server.
          </p>
        )}

        {data && data.length > 0 && (
          <ScrollArea className="max-h-[60vh] w-full">
            <div className="w-full divide-y overflow-hidden pr-3">
              {data.map((tool, i) => {
                const name = tool.name ?? tool.title ?? "unknown"
                return (
                  <DiscoveredToolRow
                    key={`${name ?? i}`}
                    tool={tool}
                    serverKey={server.server_key}
                    isRegistered={registeredToolNames.has(name)}
                    onCreated={() => refetch()}
                  />
                )
              })}
            </div>
          </ScrollArea>
        )}
      </DialogContent>
    </Dialog>
  )
}

export function MCPServerCard({ server }: { server: MCPServerConfig }) {
  const [discoverOpen, setDiscoverOpen] = useState(false)
  const authLabel = AUTH_LABELS[server.auth?.type ?? "none"] ?? "Unknown"

  return (
    <>
      <div className="flex items-center gap-4 px-4 py-3 bg-card hover:bg-accent/30 transition-colors">
        {/* Name + key */}
        <div className="min-w-0 flex-1">
          <span className="text-sm font-medium truncate block">{server.name}</span>
          <span className="font-mono text-xs text-muted-foreground truncate block">{server.server_key}</span>
        </div>

        {/* URL */}
        <div className="hidden md:flex items-center gap-1.5 w-56 shrink-0">
          <Globe className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className="font-mono text-xs text-muted-foreground truncate">{server.url}</span>
        </div>

        {/* Auth */}
        <div className="hidden md:flex items-center shrink-0">
          <div className="w-20 flex justify-center">
            <Badge variant="outline" className="text-xs">{authLabel}</Badge>
          </div>
        </div>

        {/* Discover */}
        <Button
          size="icon"
          variant="ghost"
          className="h-7 w-7 shrink-0"
          title="Discover tools"
          onClick={() => setDiscoverOpen(true)}
        >
          <Search className="h-3.5 w-3.5" />
        </Button>
      </div>

      <DiscoverDialog
        server={server}
        open={discoverOpen}
        onOpenChange={setDiscoverOpen}
      />
    </>
  )
}
