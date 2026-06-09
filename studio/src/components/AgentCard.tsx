import { Eye, Pencil, Play, Trash2 } from "lucide-react"
import type { AgentConfig } from "@/lib/types"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { MoreVertical } from "lucide-react"

export function AgentCard({
  agent,
  onRun,
  onEdit,
  onDiscover,
  onDelete,
}: {
  agent: AgentConfig
  onRun: (agent: AgentConfig) => void
  onEdit: (agent: AgentConfig) => void
  onDiscover: (agent: AgentConfig) => void
  onDelete: (agent: AgentConfig) => void
}) {
  return (
    <div className="flex items-center gap-4 px-4 py-3 bg-card hover:bg-accent/30 transition-colors">
      {/* Name + key */}
      <div className="min-w-0 flex-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="text-sm font-medium truncate block cursor-default">{agent.name}</span>
          </TooltipTrigger>
          {agent.description && (
            <TooltipContent side="top" className="max-w-xs text-xs">{agent.description}</TooltipContent>
          )}
        </Tooltip>
        <span className="font-mono text-xs text-muted-foreground truncate block">{agent.agent_key}</span>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1 shrink-0">
        <Button size="sm" onClick={() => onRun(agent)}>
          <Play className="h-3.5 w-3.5" />
          Run
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-7 w-7">
              <MoreVertical className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => onEdit(agent)}>
              <Pencil className="h-4 w-4" />
              Edit
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onDiscover(agent)}>
              <Eye className="h-4 w-4" />
              Discover
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem variant="destructive" onClick={() => onDelete(agent)}>
              <Trash2 className="h-4 w-4" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  )
}
