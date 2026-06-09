import { createContext, useContext, useEffect, useState } from "react"

type Theme = "dark" | "light" | "system"

interface ThemeProviderState {
  theme: Theme
  resolved: "dark" | "light"
  setTheme: (theme: Theme) => void
  toggle: () => void
}

const ThemeProviderContext = createContext<ThemeProviderState | null>(null)
const STORAGE_KEY = "solidcue-theme"

function systemPrefersDark() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  // Dark-first: default to dark when nothing stored (STUDIO.md §3.4, decision H).
  const [theme, setThemeState] = useState<Theme>(
    () => (localStorage.getItem(STORAGE_KEY) as Theme) || "dark",
  )
  const [resolved, setResolved] = useState<"dark" | "light">("dark")

  useEffect(() => {
    const root = window.document.documentElement
    const apply = () => {
      const next = theme === "system" ? (systemPrefersDark() ? "dark" : "light") : theme
      root.classList.toggle("dark", next === "dark")
      setResolved(next)
    }
    apply()
    if (theme === "system") {
      const mq = window.matchMedia("(prefers-color-scheme: dark)")
      mq.addEventListener("change", apply)
      return () => mq.removeEventListener("change", apply)
    }
  }, [theme])

  const setTheme = (t: Theme) => {
    localStorage.setItem(STORAGE_KEY, t)
    setThemeState(t)
  }

  return (
    <ThemeProviderContext.Provider
      value={{
        theme,
        resolved,
        setTheme,
        toggle: () => setTheme(resolved === "dark" ? "light" : "dark"),
      }}
    >
      {children}
    </ThemeProviderContext.Provider>
  )
}

export function useTheme() {
  const ctx = useContext(ThemeProviderContext)
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider")
  return ctx
}
