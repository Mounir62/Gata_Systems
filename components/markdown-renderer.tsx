"use client"

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
import rehypeHighlight from "rehype-highlight"
import { cn } from "@/lib/utils"

interface MarkdownRendererProps {
  content: string
  className?: string
}

export function MarkdownRenderer({ content, className }: MarkdownRendererProps) {
  return (
    <div className={cn("prose prose-sm dark:prose-invert max-w-none", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={{
          // Headings
          h1: ({ children }) => (
            <h1 className="text-xl font-bold mt-6 mb-3 text-foreground">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-lg font-semibold mt-5 mb-2 text-foreground border-b border-border pb-1">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-base font-semibold mt-4 mb-2 text-foreground">{children}</h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-sm font-semibold mt-3 mb-1 text-foreground">{children}</h4>
          ),
          // Paragraphs
          p: ({ children }) => (
            <p className="my-2 leading-relaxed text-foreground">{children}</p>
          ),
          // Lists
          ul: ({ children }) => (
            <ul className="my-2 ml-4 list-disc space-y-1">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="my-2 ml-4 list-decimal space-y-1">{children}</ol>
          ),
          li: ({ children }) => (
            <li className="text-foreground">{children}</li>
          ),
          // Code blocks
          code: ({ className, children, ...props }) => {
            const isInline = !className
            if (isInline) {
              return (
                <code
                  className="px-1.5 py-0.5 rounded bg-muted text-sm font-mono text-foreground"
                  {...props}
                >
                  {children}
                </code>
              )
            }
            return (
              <code
                className={cn(
                  "block overflow-x-auto rounded-lg bg-muted p-3 text-sm font-mono",
                  className
                )}
                {...props}
              >
                {children}
              </code>
            )
          },
          pre: ({ children }) => (
            <pre className="my-3 overflow-x-auto rounded-lg bg-muted p-0">
              {children}
            </pre>
          ),
          // Tables
          table: ({ children }) => (
            <div className="my-4 overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-sm">{children}</table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-muted/50">{children}</thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-border">{children}</tbody>
          ),
          tr: ({ children }) => (
            <tr className="border-b border-border last:border-0">{children}</tr>
          ),
          th: ({ children }) => (
            <th className="px-3 py-2 text-left font-semibold text-foreground">{children}</th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-2 text-foreground">{children}</td>
          ),
          // Blockquotes
          blockquote: ({ children }) => (
            <blockquote className="my-3 border-l-4 border-primary/50 pl-4 italic text-muted-foreground">
              {children}
            </blockquote>
          ),
          // Horizontal rule
          hr: () => <hr className="my-6 border-border" />,
          // Links
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline hover:text-primary/80"
            >
              {children}
            </a>
          ),
          // Strong and emphasis
          strong: ({ children }) => (
            <strong className="font-semibold text-foreground">{children}</strong>
          ),
          em: ({ children }) => (
            <em className="italic">{children}</em>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
