import type { ProviderType, RoleForm as RoleFormData } from "@/lib/agent-config"
import { PROVIDER_META } from "@/lib/agent-config"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

export function RoleForm({
  value,
  onChange,
}: {
  value: RoleFormData
  onChange: (next: RoleFormData) => void
}) {
  const meta = PROVIDER_META[value.provider_type]
  const set = (patch: Partial<RoleFormData>) => onChange({ ...value, ...patch })

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <div className="space-y-1.5">
        <Label>Provider type</Label>
        <Select
          value={value.provider_type}
          onValueChange={(v) => set({ provider_type: v as ProviderType })}
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(PROVIDER_META).map(([key, m]) => (
              <SelectItem key={key} value={key}>
                {m.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-1.5">
        <Label>Base URL</Label>
        <Input
          value={meta.needsBaseUrl ? value.base_url : meta.defaultBaseUrl ?? ""}
          onChange={(e) => set({ base_url: e.target.value })}
          disabled={!meta.needsBaseUrl}
          placeholder={meta.needsBaseUrl ? "https://api.example.com/v1" : ""}
        />
      </div>

      <div className="space-y-1.5">
        <Label>Model</Label>
        <Input
          value={value.model}
          onChange={(e) => set({ model: e.target.value })}
          placeholder="model-name"
        />
      </div>

      <div className="space-y-1.5">
        <Label>Temperature</Label>
        <Input
          type="number"
          step="0.1"
          min="0"
          max="2"
          value={value.temperature}
          onChange={(e) => set({ temperature: e.target.value })}
          placeholder="0.3"
        />
      </div>

      <div className="space-y-1.5 sm:col-span-2">
        <Label>API key</Label>
        <Input
          type="password"
          value={value.api_key}
          onChange={(e) => set({ api_key: e.target.value })}
          placeholder="sk-…  (stored to an env file by the backend)"
          autoComplete="off"
        />
      </div>
    </div>
  )
}
