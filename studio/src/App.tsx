import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ThemeProvider } from "@/components/theme-provider"
import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import { AppLayout } from "@/components/layout/AppLayout"
import { AgentsPage } from "@/pages/AgentsPage"
import { CreateAgentPage } from "@/pages/CreateAgentPage"
import { MCPPage } from "@/pages/MCPPage"
import { ToolsPage } from "@/pages/ToolsPage"
import { SessionsPage } from "@/pages/SessionsPage"
import { ProfilePage } from "@/pages/ProfilePage"

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
})

export default function App() {
  return (
    <ThemeProvider>
      <TooltipProvider delayDuration={50}>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Routes>
            <Route element={<AppLayout />}>
              <Route index element={<Navigate to="/agents" replace />} />
              <Route path="/agents" element={<AgentsPage />} />
              <Route path="/agents/new" element={<CreateAgentPage />} />
              <Route path="/mcp" element={<MCPPage />} />
              <Route path="/tools" element={<ToolsPage />} />
              <Route path="/sessions" element={<SessionsPage />} />
              <Route path="/profile" element={<ProfilePage />} />
            </Route>
            <Route path="*" element={<Navigate to="/agents" replace />} />
          </Routes>
        </BrowserRouter>
        <Toaster />
      </QueryClientProvider>
      </TooltipProvider>
    </ThemeProvider>
  )
}
