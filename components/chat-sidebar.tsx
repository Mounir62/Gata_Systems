"use client"

import React from "react"

import { useState, useCallback, useEffect, useRef } from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { MessageSquare, Plus, ChevronLeft, Menu, Trash2 } from "lucide-react"

interface Conversation {
  id: string
  title: string
  lastMessage: string
  timestamp: Date
}

interface ChatSidebarProps {
  conversations: Conversation[]
  activeConversation: string | null
  onSelectConversation: (id: string) => void
  onNewConversation: () => void
  onDeleteConversation: (id: string) => void
  isCollapsed: boolean
  onToggleCollapse: () => void
}

const MIN_WIDTH = 200
const MAX_WIDTH = 400
const COLLAPSED_WIDTH = 64

export function ChatSidebar({
  conversations,
  activeConversation,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  isCollapsed,
  onToggleCollapse,
}: ChatSidebarProps) {
  const [sidebarWidth, setSidebarWidth] = useState(288) // 72 * 4 = 288px (w-72)
  const [isResizing, setIsResizing] = useState(false)
  const sidebarRef = useRef<HTMLDivElement>(null)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizing(true)
  }, [])

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing) return
      const newWidth = e.clientX
      if (newWidth >= MIN_WIDTH && newWidth <= MAX_WIDTH) {
        setSidebarWidth(newWidth)
      }
    },
    [isResizing]
  )

  const handleMouseUp = useCallback(() => {
    setIsResizing(false)
  }, [])

  useEffect(() => {
    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove)
      document.addEventListener("mouseup", handleMouseUp)
      document.body.style.cursor = "col-resize"
      document.body.style.userSelect = "none"
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }
  }, [isResizing, handleMouseMove, handleMouseUp])

  const currentWidth = isCollapsed ? COLLAPSED_WIDTH : sidebarWidth

  return (
    <div
      ref={sidebarRef}
      style={{ width: currentWidth }}
      className={cn(
        "relative flex h-full flex-col border-r border-border overflow-hidden",
        "bg-background md:bg-sidebar-background",
        !isResizing && "transition-all duration-300 ease-in-out",
        "md:relative"
      )}
    >
      {/* Header */}
      <div className="flex h-14 sm:h-16 items-center justify-between border-b border-sidebar-border px-3 sm:px-4">
        {!isCollapsed && (
          <div className="flex items-center gap-2">
            <img
              src="/images/image.png"
              alt="GATA Logo"
              className="h-6 sm:h-8 w-auto"
            />
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleCollapse}
          className="ml-auto text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground active:scale-95 touch-manipulation"
        >
          {isCollapsed ? (
            <Menu className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* New Chat Button */}
      <div className="p-2 sm:p-3">
        <Button
          onClick={onNewConversation}
          className={cn(
            "w-full justify-start gap-2 bg-primary text-primary-foreground hover:bg-primary/90 active:scale-95 touch-manipulation h-10 sm:h-auto",
            isCollapsed && "justify-center px-2"
          )}
        >
          <Plus className="h-4 w-4" />
          {!isCollapsed && <span>New Chat</span>}
        </Button>
      </div>

      {/* Conversations List */}
      <ScrollArea className="flex-1">
        <div className="space-y-1 px-2 sm:px-3">
          {conversations.map((conversation) => (
            <div
              key={conversation.id}
              className={cn(
                "group relative h-10 sm:h-11 w-full rounded-md transition-colors",
                activeConversation === conversation.id
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "hover:bg-sidebar-accent/50 text-sidebar-foreground"
              )}
            >
              <button
                onClick={() => onSelectConversation(conversation.id)}
                className={cn(
                  "absolute inset-0 flex items-center gap-2 sm:gap-3 pl-2 sm:pl-3 pr-8 sm:pr-10 text-left text-sm sm:text-base font-medium active:scale-95 touch-manipulation",
                  isCollapsed && "justify-center pr-2 sm:pr-3 pl-2 sm:pl-3"
                )}
              >
                <MessageSquare className="h-4 w-4 sm:h-5 sm:w-5 shrink-0" />
                {!isCollapsed && (
                  <span className="truncate text-sm sm:text-base">{conversation.title}</span>
                )}
              </button>
              {!isCollapsed && (
                <div className="absolute right-1 sm:right-2 top-1/2 -translate-y-1/2 flex items-center gap-1 sm:gap-2 opacity-0 group-hover:opacity-100 transition-all">
                  <div className="h-5 w-px bg-border" />
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteConversation(conversation.id)
                    }}
                    className="p-1 rounded text-muted-foreground hover:text-destructive transition-colors touch-manipulation active:scale-90"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </ScrollArea>

      {/* Footer */}
      {!isCollapsed && (
        <div className="border-t border-sidebar-border p-4">
          <p className="text-xs text-muted-foreground text-center">
            Powered by GATA AI
          </p>
        </div>
      )}

      {/* Resize Handle */}
      {!isCollapsed && (
        <div
          onMouseDown={handleMouseDown}
          className={cn(
            "absolute right-0 top-0 h-full w-1 cursor-col-resize hover:bg-primary/50 transition-colors",
            isResizing && "bg-primary"
          )}
        />
      )}
    </div>
  )
}
