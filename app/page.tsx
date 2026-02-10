"use client"

import * as React from "react"
import { cn } from "@/lib/utils"
import { ThemeProvider } from "@/components/theme-provider"
import { ChatSidebar } from "@/components/chat-sidebar"
import { ChatInterface } from "@/components/chat-interface"
import { ChatHeader } from "@/components/chat-header"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

interface Conversation {
  id: string
  title: string
  lastMessage: string
  timestamp: Date
  messages: Message[]
}

interface OllamaModel {
  name: string
  modified_at: string
  size: number
}

export default function ChatPage() {
  const [conversations, setConversations] = React.useState<Conversation[]>([
    {
      id: "1",
      title: "New Chat",
      lastMessage: "Start a new conversation",
      timestamp: new Date(),
      messages: [],
    },
  ])
  const [activeConversation, setActiveConversation] = React.useState<string | null>("1")
  const [isLoading, setIsLoading] = React.useState(false)
  const [isSidebarCollapsed, setIsSidebarCollapsed] = React.useState(true)
  const [selectedModel, setSelectedModel] = React.useState<string>("")
  const [availableModels, setAvailableModels] = React.useState<OllamaModel[]>([])
  const [modelsLoading, setModelsLoading] = React.useState(true)
  const [streamingContent, setStreamingContent] = React.useState<string>("")
  const [selectedDocument, setSelectedDocument] = React.useState<string | null>(null)
  const [complexity, setComplexity] = React.useState<string>("balanced")

  // Fetch available models on mount
  React.useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch("/api/ollama/models")
        if (response.ok) {
          const data = await response.json()
          setAvailableModels(data.models || [])
          if (data.models?.length > 0) {
            setSelectedModel(data.models[0].name)
          }
        }
      } catch (error) {
        console.error("Failed to fetch models:", error)
      } finally {
        setModelsLoading(false)
      }
    }
    fetchModels()
  }, [])

  const currentConversation = conversations.find((c) => c.id === activeConversation)
  const messages = currentConversation?.messages || []

  const handleSendMessage = async (content: string) => {
    if (!activeConversation || !selectedModel) return

    // Capture the conversation ID to prevent flickering
    const conversationId = activeConversation
    const conversation = conversations.find((c) => c.id === conversationId)
    if (!conversation) return

    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: "user",
      content,
      timestamp: new Date(),
    }

    // Build message history including the new user message
    const messageHistory = [
      ...conversation.messages.map((m) => ({
        role: m.role,
        content: m.content,
      })),
      { role: "user", content },
    ]

    // Add user message
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === conversationId
          ? {
              ...conv,
              messages: [...conv.messages, userMessage],
              lastMessage: content,
              timestamp: new Date(),
              title: conv.messages.length === 0 ? content.slice(0, 30) + (content.length > 30 ? "..." : "") : conv.title,
            }
          : conv
      )
    )

    // Start streaming response from Ollama
    setIsLoading(true)
    setStreamingContent("")

    try {
      const currentMessages = [
        ...(currentConversation?.messages || []).map((m) => ({
          role: m.role,
          content: m.content,
        })),
        { role: "user", content },
      ]

      const response = await fetch("/api/ollama", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: selectedModel,
          messages: currentMessages,
          selectedDocument,
          complexity,
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to get response from Ollama")
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let fullContent = ""

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value)
          const lines = chunk.split("\n")

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = line.slice(6)
              if (data === "[DONE]") continue
              try {
                const json = JSON.parse(data)
                if (json.content) {
                  fullContent += json.content
                  setStreamingContent(fullContent)
                }
              } catch {
                // Skip invalid JSON
              }
            }
          }
        }
      }

      // Add the complete assistant message
      const assistantMessage: Message = {
        id: `msg-${Date.now() + 1}`,
        role: "assistant",
        content: fullContent || "Sorry, I couldn't generate a response.",
        timestamp: new Date(),
      }

      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === conversationId
            ? {
                ...conv,
                messages: [...conv.messages, assistantMessage],
                lastMessage: assistantMessage.content.slice(0, 50) + "...",
              }
            : conv
        )
      )
    } catch (error) {
      console.error("Error:", error)
      // Add error message
      const errorMessage: Message = {
        id: `msg-${Date.now() + 1}`,
        role: "assistant",
        content: "Sorry, I couldn't connect to Ollama. Please make sure Ollama is running locally.",
        timestamp: new Date(),
      }

      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === conversationId
            ? {
                ...conv,
                messages: [...conv.messages, errorMessage],
                lastMessage: errorMessage.content.slice(0, 50) + "...",
              }
            : conv
        )
      )
    } finally {
      setIsLoading(false)
      setStreamingContent("")
    }
  }

  const handleNewConversation = () => {
    const newId = `conv-${Date.now()}`
    const newConversation: Conversation = {
      id: newId,
      title: "New Chat",
      lastMessage: "Start a new conversation",
      timestamp: new Date(),
      messages: [],
    }
    setConversations((prev) => [newConversation, ...prev])
    setActiveConversation(newId)
  }

  const handleDeleteConversation = (id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id))
    if (activeConversation === id) {
      const remaining = conversations.filter((c) => c.id !== id)
      setActiveConversation(remaining[0]?.id || null)
    }
  }

  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
      <div className="flex h-screen overflow-hidden bg-background">
        {/* Sidebar - hidden on mobile by default, overlay on mobile when open */}
        <div className={cn(
          "fixed inset-y-0 left-0 z-50 md:relative md:z-auto",
          isSidebarCollapsed && "hidden md:block"
        )}>
          <ChatSidebar
            conversations={conversations}
            activeConversation={activeConversation}
            onSelectConversation={(id) => {
              setActiveConversation(id)
              // Auto-close sidebar on mobile after selection
              if (window.innerWidth < 768) {
                setIsSidebarCollapsed(true)
              }
            }}
            onNewConversation={() => {
              handleNewConversation()
              // Auto-close sidebar on mobile after creating new chat
              if (window.innerWidth < 768) {
                setIsSidebarCollapsed(true)
              }
            }}
            onDeleteConversation={handleDeleteConversation}
            isCollapsed={isSidebarCollapsed}
            onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
          />
        </div>

        {/* Backdrop for mobile sidebar */}
        {!isSidebarCollapsed && (
          <div
            className="fixed inset-0 z-40 bg-black/50 md:hidden"
            onClick={() => setIsSidebarCollapsed(true)}
          />
        )}

        {/* Main Content */}
        <div className="flex flex-1 flex-col overflow-hidden">
          <ChatHeader
            onToggleSidebar={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            isSidebarCollapsed={isSidebarCollapsed}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            availableModels={availableModels}
            modelsLoading={modelsLoading}
            complexity={complexity}
            onComplexityChange={setComplexity}
          />
          <main className="flex-1 overflow-hidden">
            <ChatInterface
              messages={messages}
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              streamingContent={streamingContent}
              selectedDocument={selectedDocument}
              onDocumentChange={setSelectedDocument}
              complexity={complexity}
              onComplexityChange={setComplexity}
            />
          </main>
        </div>
      </div>
    </ThemeProvider>
  )
}

