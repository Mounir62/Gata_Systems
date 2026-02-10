"use client"

import * as React from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { MarkdownRenderer } from "@/components/markdown-renderer"
import { Send, Bot, User, Sparkles, Loader2, Copy, Check, Gauge } from "lucide-react"
import { PdfUpload } from "@/components/pdf-upload"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import "katex/dist/katex.min.css"
import "highlight.js/styles/github-dark.css"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

interface ChatInterfaceProps {
  messages: Message[]
  onSendMessage: (content: string) => void
  isLoading?: boolean
  streamingContent?: string
  selectedDocument: string | null
  onDocumentChange: (filename: string | null) => void
  complexity: string
  onComplexityChange: (complexity: string) => void
}

export function ChatInterface({
  messages,
  onSendMessage,
  isLoading = false,
  streamingContent = "",
  selectedDocument,
  onDocumentChange,
  complexity,
  onComplexityChange,
}: ChatInterfaceProps) {
  const [input, setInput] = React.useState("")
  const scrollRef = React.useRef<HTMLDivElement>(null)
  const inputRef = React.useRef<HTMLTextAreaElement>(null)
  const [copiedId, setCopiedId] = React.useState<string | null>(null)

  const copyToClipboard = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedId(id)
      setTimeout(() => setCopiedId(null), 2000)
    } catch (err) {
      console.error("Failed to copy:", err)
    }
  }

  React.useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, streamingContent])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim())
      setInput("")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* Messages Area */}
      <ScrollArea ref={scrollRef} className="flex-1 px-2 sm:px-4">
        <div className="mx-auto max-w-3xl py-4 sm:py-8">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 sm:py-16 px-3 sm:px-4">
              <div className="mb-3 sm:mb-6 flex h-14 w-14 sm:h-16 sm:w-16 items-center justify-center rounded-2xl bg-primary/10">
                <Sparkles className="h-7 w-7 sm:h-8 sm:w-8 text-primary" />
              </div>
              <h2 className="mb-1 sm:mb-2 text-xl sm:text-2xl md:text-3xl font-bold text-foreground text-center">
                Welcome to GATA
              </h2>
              <p className="mb-4 sm:mb-6 md:mb-8 max-w-lg text-center text-sm sm:text-base text-muted-foreground px-2 sm:px-4 leading-relaxed">
                Your intelligent pump systems assistant. Ask me about maintenance, troubleshooting, or optimization.
              </p>
              <div className="grid gap-2 sm:gap-3 grid-cols-1 sm:grid-cols-2 w-full max-w-2xl px-2 sm:px-4">
                {[
                  "Provide a detailed maintenance plan",
                  "Troubleshoot pump performance issues",
                  "Optimize energy efficiency",
                  "Recommend spare parts checklist",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => onSendMessage(suggestion)}
                    className="rounded-xl border border-border bg-card px-3 sm:px-4 py-3 sm:py-3.5 text-left text-xs sm:text-sm text-card-foreground transition-all hover:bg-accent hover:text-accent-foreground hover:shadow-md active:scale-[0.98] touch-manipulation font-medium"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-4 sm:space-y-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex gap-2 sm:gap-4",
                    message.role === "user" ? "justify-end" : "justify-start"
                  )}
                >
                  {message.role === "assistant" && (
                    <div className="flex h-7 w-7 sm:h-9 sm:w-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                      <Bot className="h-4 w-4 sm:h-5 sm:w-5" />
                    </div>
                  )}
                  <div
                    className={cn(
                      "max-w-[85%] sm:max-w-[80%] rounded-2xl px-3 sm:px-4 py-2 sm:py-3",
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-card border border-border text-card-foreground"
                    )}
                  >
                    {message.role === "assistant" ? (
                      <>
                        <MarkdownRenderer content={message.content} className="text-sm" />
                        <div className="mt-2 flex items-center justify-between gap-2">
                          <p className="text-xs text-muted-foreground">
                            {message.timestamp.toLocaleTimeString([], {
                              hour: "2-digit",
                              minute: "2-digit",
                            })}
                          </p>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => copyToClipboard(message.content, message.id)}
                            className="h-6 px-2 text-xs hover:bg-accent"
                          >
                            {copiedId === message.id ? (
                              <>
                                <Check className="h-3 w-3 mr-1" />
                                Copied
                              </>
                            ) : (
                              <>
                                <Copy className="h-3 w-3 mr-1" />
                                Copy
                              </>
                            )}
                          </Button>
                        </div>
                      </>
                    ) : (
                      <>
                        <p className="whitespace-pre-wrap text-sm leading-relaxed">
                          {message.content}
                        </p>
                        <p className="mt-2 text-xs text-primary-foreground/70">
                          {message.timestamp.toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </p>
                      </>
                    )}
                  </div>
                  {message.role === "user" && (
                    <div className="flex h-7 w-7 sm:h-9 sm:w-9 shrink-0 items-center justify-center rounded-full bg-secondary text-secondary-foreground">
                      <User className="h-4 w-4 sm:h-5 sm:w-5" />
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="flex gap-2 sm:gap-4">
                  <div className="flex h-7 w-7 sm:h-9 sm:w-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                    <Bot className="h-4 w-4 sm:h-5 sm:w-5" />
                  </div>
                  <div className="max-w-[85%] sm:max-w-[80%] rounded-2xl border border-border bg-card px-3 sm:px-4 py-2 sm:py-3 text-card-foreground">
                    {streamingContent ? (
                      <div className="text-sm">
                        <MarkdownRenderer content={streamingContent} className="text-sm" />
                        <span className="inline-block w-2 h-4 ml-1 bg-primary animate-pulse" />
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>Thinking...</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="border-t border-border bg-background p-2 sm:p-4 safe-bottom">
        <div className="mx-auto max-w-3xl mb-2">
          <PdfUpload
            selectedDocument={selectedDocument}
            onDocumentChange={onDocumentChange}
          />
        </div>
        <form
          onSubmit={handleSubmit}
          className="mx-auto flex max-w-3xl items-end gap-2 sm:gap-3"
        >
          <div className="relative flex-1">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message..."
              rows={1}
              className="w-full resize-none rounded-xl border border-input bg-card px-3 sm:px-4 py-2 sm:py-3 text-sm text-card-foreground placeholder:text-muted-foreground focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring touch-manipulation"
              style={{
                minHeight: "44px",
                maxHeight: "120px",
              }}
            />
          </div>
          <Button
            type="submit"
            size="icon"
            disabled={!input.trim() || isLoading}
            className="h-11 w-11 sm:h-12 sm:w-12 shrink-0 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 active:scale-95 touch-manipulation"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 sm:h-5 sm:w-5 animate-spin" />
            ) : (
              <Send className="h-4 w-4 sm:h-5 sm:w-5" />
            )}
          </Button>
        </form>
        <p className="mx-auto mt-2 max-w-3xl text-center text-xs text-muted-foreground px-2">
          GATA may produce inaccurate information. Consider checking important facts.
        </p>
      </div>
    </div>
  )
}
