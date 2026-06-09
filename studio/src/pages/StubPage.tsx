import { PageHeader } from "@/components/PageHeader"

export function StubPage({ title, description }: { title: string; description: string }) {
  return (
    <>
      <PageHeader title={title} description={description} />
      <div className="rounded-lg border border-dashed border-border p-12 text-center text-sm text-muted-foreground">
        Coming next in the build order (see STUDIO.md §8).
      </div>
    </>
  )
}
