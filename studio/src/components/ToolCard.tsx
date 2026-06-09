import { MoreHorizontal, Pencil } from "lucide-react"
import type { ToolConfig } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Switch } from "@/components/ui/switch"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

const TYPE_COLORS: Record<string, string> = {
  mcp: "bg-violet-500/10 text-violet-400 border-violet-500/20",
  api: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  rag: "bg-amber-500/10 text-amber-400 border-amber-500/20",
}

const RISK_COLORS: Record<string, string> = {
  low: "bg-green-500/10 text-green-400 border-green-500/20",
  medium: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  high: "bg-red-500/10 text-red-400 border-red-500/20",
}

export function ToolCard({
  tool,
  onToggle,
  onEdit,
  toggling,
}: {
  tool: ToolConfig
  onToggle: (enabled: boolean) => void
  onEdit: () => void
  toggling?: boolean
}) {
  const typeColor = TYPE_COLORS[tool.type] ?? ""
  const riskColor = RISK_COLORS[tool.approval_risk] ?? ""
  const serverKey = tool.mcp?.server_key

  return (
    <div className={cn("flex items-center gap-4 px-4 py-3 bg-card hover:bg-accent/30 transition-colors", !tool.enabled && "opacity-50")}>
      {/* Name + meta */}
      <div className="min-w-0 flex-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="text-sm font-medium truncate block cursor-default">{tool.name}</span>
          </TooltipTrigger>
          {tool.description && (
            <TooltipContent side="top" className="max-w-xs text-xs">
              {tool.description}
            </TooltipContent>
          )}
        </Tooltip>
        <span className="font-mono text-xs text-muted-foreground truncate block">{tool.tool_key}</span>
      </div>

      {/* Badges — fixed-width columns so rows align */}
      <div className="hidden md:flex items-center gap-3 shrink-0">
        <div className="w-12 flex justify-center">
          <Badge variant="outline" className={cn("text-xs", typeColor)}>{tool.type}</Badge>
        </div>
        <div className="w-16 flex justify-center">
          <Badge variant="outline" className={cn("text-xs capitalize", riskColor)}>{tool.approval_risk}</Badge>
        </div>
        <div className="w-28 flex justify-start">
          {serverKey ? (
            <Badge variant="outline" className="text-xs font-mono truncate max-w-full">{serverKey}</Badge>
          ) : (
            <span className="text-xs text-muted-foreground/40">—</span>
          )}
        </div>
      </div>

      {/* Toggle + menu */}
      <div className="flex items-center gap-2 shrink-0">
        <Switch
          checked={tool.enabled}
          onCheckedChange={onToggle}
          disabled={toggling}
          aria-label={`Toggle ${tool.name}`}
        />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-7 w-7">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={onEdit}>
              <Pencil className="h-3.5 w-3.5 mr-2" />
              Edit
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  )
}
