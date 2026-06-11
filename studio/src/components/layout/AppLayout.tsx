import { useCallback, useRef, useState } from "react"
import { Outlet, useLocation } from "react-router-dom"
import { cn } from "@/lib/utils"
import { Sidebar } from "./Sidebar"

const MIN_WIDTH = 160
const MAX_WIDTH = 400
const DEFAULT_WIDTH = 224 // w-56

export function AppLayout() {
  const location = useLocation()
  const [sidebarWidth, setSidebarWidth] = useState(DEFAULT_WIDTH)
  const dragging = useRef(false)
  const isSessionsPage = location.pathname === "/sessions"

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = true
    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"

    const onMouseMove = (ev: MouseEvent) => {
      if (!dragging.current) return
      setSidebarWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, ev.clientX)))
    }
    const onMouseUp = () => {
      dragging.current = false
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
      document.removeEventListener("mousemove", onMouseMove)
      document.removeEventListener("mouseup", onMouseUp)
    }
    document.addEventListener("mousemove", onMouseMove)
    document.addEventListener("mouseup", onMouseUp)
  }, [])

  return (
    <div className="flex h-svh w-full overflow-hidden">
      <div style={{ width: sidebarWidth }} className="flex shrink-0 flex-col overflow-hidden border-r border-border bg-card">
        <Sidebar />
      </div>
      <div
        onMouseDown={onMouseDown}
        className="w-1 shrink-0 cursor-col-resize hover:bg-primary/40 active:bg-primary/60 transition-colors"
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <main
          className={cn(
            "flex-1 overflow-y-auto",
            isSessionsPage ? "min-h-0 p-0" : "p-6",
          )}
        >
          <Outlet />
        </main>
      </div>
    </div>
  )
}
