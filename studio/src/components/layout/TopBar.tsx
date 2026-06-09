import { useQuery } from "@tanstack/react-query"
import { Moon, Sun } from "lucide-react"
import { api } from "@/lib/api"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { useTheme } from "@/components/theme-provider"

export function TopBar() {
  const { resolved, toggle } = useTheme()
  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15000,
    retry: false,
  })

  const ok = health.isSuccess && health.data?.status === "ok"

  return (
    <header className="flex h-14 shrink-0 items-center justify-end gap-4 border-b border-border px-6">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span
          className={cn(
            "h-2 w-2 rounded-full",
            health.isLoading ? "bg-warning" : ok ? "bg-success" : "bg-destructive",
          )}
        />
        {health.isLoading ? "connecting" : ok ? "API ok" : "API down"}
      </div>
      <Button variant="ghost" size="icon" onClick={toggle} aria-label="Toggle theme">
        {resolved === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
      </Button>
    </header>
  )
}
