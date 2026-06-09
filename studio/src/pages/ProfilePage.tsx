import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Loader2, Save } from "lucide-react"
import { toast } from "sonner"
import { api, ApiError } from "@/lib/api"
import type { UserProfileConfig } from "@/lib/types"
import { PageHeader } from "@/components/PageHeader"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"

const EMPTY: UserProfileConfig = {
  display_name: null,
  location: null,
  timezone: null,
  personality: null,
  preferences: {},
}

export function ProfilePage() {
  const qc = useQueryClient()
  const { data, isLoading } = useQuery({ queryKey: ["profile"], queryFn: api.getProfile })

  const [form, setForm] = useState<UserProfileConfig>(EMPTY)
  const [dirty, setDirty] = useState(false)

  useEffect(() => {
    if (data) {
      setForm(data)
      setDirty(false)
    }
  }, [data])

  function set(patch: Partial<UserProfileConfig>) {
    setForm((prev) => ({ ...prev, ...patch }))
    setDirty(true)
  }

  const save = useMutation({
    mutationFn: () => api.updateProfile(form),
    onSuccess: (updated) => {
      toast.success("Profile saved")
      qc.setQueryData(["profile"], updated)
      setDirty(false)
    },
    onError: (e: ApiError) => toast.error(e.message),
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <PageHeader
        title="Profile"
        description="Your identity and preferences used by agents."
        action={
          <Button
            disabled={!dirty || save.isPending}
            onClick={() => save.mutate()}
          >
            {save.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            Save
          </Button>
        }
      />

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Identity</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <Label>Display name</Label>
            <Input
              value={form.display_name ?? ""}
              onChange={(e) => set({ display_name: e.target.value || null })}
              placeholder="Your name"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <Label>Location</Label>
              <Input
                value={form.location ?? ""}
                onChange={(e) => set({ location: e.target.value || null })}
                placeholder="Singapore"
              />
            </div>
            <div className="space-y-1.5">
              <Label>Timezone</Label>
              <Input
                value={form.timezone ?? ""}
                onChange={(e) => set({ timezone: e.target.value || null })}
                placeholder="Asia/Singapore"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Personality</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-1.5">
            <Label>How should agents address you?</Label>
            <Textarea
              value={form.personality ?? ""}
              onChange={(e) => set({ personality: e.target.value || null })}
              placeholder="e.g. Direct and concise. Prefer bullet points over prose. Skip pleasantries."
              rows={4}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
