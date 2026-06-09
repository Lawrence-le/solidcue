import { useMemo, useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import rehypeHighlight from "rehype-highlight"
import { Check, Copy } from "lucide-react"
import { cn } from "@/lib/utils"

// Borrowed from apicue: auto-closes unclosed code fences during streaming
// so the markdown parser never sees an unterminated fence block.
function normalize(input: string): string {
  const text = input.replace(/\r\n/g, "\n")
  const fences = (text.match(/```/g) ?? []).length
  return fences % 2 !== 0 ? text + "\n```" : text
}

function extractText(node: React.ReactNode): string {
  if (node == null || typeof node === "boolean") return ""
  if (typeof node === "string" || typeof node === "number") return String(node)
  if (Array.isArray(node)) return node.map(extractText).join("")
  if (typeof node === "object" && node !== null && "props" in node) {
    const el = node as React.ReactElement<{ children?: React.ReactNode }>
    return extractText(el.props.children)
  }
  return ""
}

function CopyButton({ code }: { code: string }) {
  const [copied, setCopied] = useState(false)
  function handleCopy() {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }
  return (
    <button
      type="button"
      onClick={handleCopy}
      className="flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-muted-foreground hover:bg-white/10 hover:text-foreground transition-colors"
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? "Copied" : "Copy"}
    </button>
  )
}

export function MarkdownContent({
  content,
  className,
}: {
  content: string
  className?: string
}) {
  const [codeBlockCounter] = useState(() => ({ n: 0 }))
  codeBlockCounter.n = 0

  const components = useMemo(
    () => ({
      // Paragraphs
      p: ({ children }: React.HTMLAttributes<HTMLParagraphElement>) => (
        <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>
      ),
      // Headings
      h1: ({ children }: React.HTMLAttributes<HTMLHeadingElement>) => (
        <h1 className="mb-3 mt-5 text-xl font-semibold first:mt-0">{children}</h1>
      ),
      h2: ({ children }: React.HTMLAttributes<HTMLHeadingElement>) => (
        <h2 className="mb-2 mt-4 text-lg font-semibold first:mt-0">{children}</h2>
      ),
      h3: ({ children }: React.HTMLAttributes<HTMLHeadingElement>) => (
        <h3 className="mb-2 mt-3 text-base font-semibold first:mt-0">{children}</h3>
      ),
      // Lists
      ul: ({ children }: React.HTMLAttributes<HTMLUListElement>) => (
        <ul className="mb-3 list-disc space-y-1 pl-5">{children}</ul>
      ),
      ol: ({ children }: React.HTMLAttributes<HTMLOListElement>) => (
        <ol className="mb-3 list-decimal space-y-1 pl-5">{children}</ol>
      ),
      li: ({ children }: React.HTMLAttributes<HTMLLIElement>) => (
        <li className="leading-relaxed">{children}</li>
      ),
      // Blockquote
      blockquote: ({ children }: React.HTMLAttributes<HTMLQuoteElement>) => (
        <blockquote className="my-3 border-l-2 border-primary/40 pl-4 italic text-muted-foreground">
          {children}
        </blockquote>
      ),
      // Inline formatting
      strong: ({ children }: React.HTMLAttributes<HTMLElement>) => (
        <strong className="font-semibold">{children}</strong>
      ),
      em: ({ children }: React.HTMLAttributes<HTMLElement>) => (
        <em className="italic">{children}</em>
      ),
      del: ({ children }: React.HTMLAttributes<HTMLElement>) => (
        <del className="line-through opacity-70">{children}</del>
      ),
      // Links
      a: ({ href, children }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary underline underline-offset-2 hover:opacity-80"
        >
          {children}
        </a>
      ),
      // Horizontal rule
      hr: () => <hr className="my-4 border-border" />,
      // Tables
      table: ({ children }: React.HTMLAttributes<HTMLTableElement>) => (
        <div className="my-3 overflow-x-auto rounded-md border border-border">
          <table className="min-w-full text-sm">{children}</table>
        </div>
      ),
      thead: ({ children }: React.HTMLAttributes<HTMLTableSectionElement>) => (
        <thead className="bg-muted/50">{children}</thead>
      ),
      th: ({ children }: React.HTMLAttributes<HTMLTableCellElement>) => (
        <th className="border-r border-border px-3 py-2 text-left font-medium last:border-r-0">
          {children}
        </th>
      ),
      td: ({ children }: React.HTMLAttributes<HTMLTableCellElement>) => (
        <td className="border-r border-t border-border px-3 py-2 last:border-r-0">{children}</td>
      ),
      // Code — inline and block
      code: ({
        className: cls,
        children,
        ...rest
      }: React.HTMLAttributes<HTMLElement>) => {
        const isBlock = cls?.startsWith("language-") || String(children).includes("\n")
        if (!isBlock) {
          return (
            <code
              className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs"
              {...rest}
            >
              {children}
            </code>
          )
        }
        const lang = cls?.replace("language-", "") ?? ""
        const id = `cb-${++codeBlockCounter.n}`
        const codeText = extractText(children)
        return (
          <div className="my-3 overflow-hidden rounded-md border border-border bg-zinc-900 dark:bg-zinc-950">
            <div className="flex items-center justify-between border-b border-border/60 bg-zinc-800/60 px-3 py-1.5">
              <span className="font-mono text-xs text-muted-foreground">{lang || "code"}</span>
              <CopyButton key={id} code={codeText} />
            </div>
            <pre className="overflow-x-auto p-4">
              <code className={cn("font-mono text-xs leading-relaxed", cls)} {...rest}>
                {children}
              </code>
            </pre>
          </div>
        )
      },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

  const normalized = normalize(content)

  return (
    <div className={cn("text-sm", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={components}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  )
}
