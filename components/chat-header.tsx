"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { useTheme } from "@/components/theme-provider"
import { Moon, Sun, Menu, Settings, HelpCircle, ChevronDown, Cpu, Loader2, Database, GraduationCap, Gauge, Zap, BarChart2, Layers } from "lucide-react"
import Link from "next/link"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu"
import { Badge } from "@/components/ui/badge"

interface OllamaModel {
  name: string
  modified_at: string
  size: number
}

interface ChatHeaderProps {
  onToggleSidebar: () => void
  isSidebarCollapsed: boolean
  selectedModel: string
  onModelChange: (model: string) => void
  availableModels: OllamaModel[]
  modelsLoading: boolean
  complexity: string
  onComplexityChange: (complexity: string) => void
}

export function ChatHeader({ 
  onToggleSidebar, 
  isSidebarCollapsed,
  selectedModel,
  onModelChange,
  availableModels,
  modelsLoading,
  complexity,
  onComplexityChange,
}: ChatHeaderProps) {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <header className="flex h-14 sm:h-16 items-center justify-between border-b border-border bg-background px-3 sm:px-4">
      <div className="flex items-center gap-2 sm:gap-3">
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleSidebar}
          className="md:hidden active:scale-95 touch-manipulation"
        >
          <Menu className="h-5 w-5" />
        </Button>
        <div className="flex items-center gap-2">
          <img
            src="/images/image.png"
            alt="GATA Logo"
            className="h-6 sm:h-7 w-auto"
          />
          <span className="hidden sm:inline text-sm font-medium text-muted-foreground">
            AI Assistant
          </span>
          <div className="hidden md:flex items-center ml-3 pl-3 border-l border-border">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="flex items-center gap-2 h-8 px-3 hover:bg-accent active:scale-95 touch-manipulation group"
                >
                  <div className="flex items-center gap-1.5">
                    {complexity === "simple" && <Zap className="h-3.5 w-3.5 text-blue-500" />}
                    {complexity === "balanced" && <BarChart2 className="h-3.5 w-3.5 text-green-500" />}
                    {complexity === "detailed" && <Layers className="h-3.5 w-3.5 text-purple-500" />}
                    <span className="text-xs font-medium capitalize">{complexity}</span>
                  </div>
                  <ChevronDown className="h-3 w-3 text-muted-foreground group-hover:text-foreground transition-colors" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-56">
                <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
                  Answer Complexity
                </div>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={() => onComplexityChange("simple")}
                  className={`cursor-pointer ${complexity === "simple" ? "bg-accent" : ""}`}
                >
                  <div className="flex items-start gap-3 w-full">
                    <Zap className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <div className="flex-1">
                      <div className="font-medium text-sm">Simple</div>
                      <div className="text-xs text-muted-foreground">Quick, concise answers</div>
                    </div>
                  </div>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => onComplexityChange("balanced")}
                  className={`cursor-pointer ${complexity === "balanced" ? "bg-accent" : ""}`}
                >
                  <div className="flex items-start gap-3 w-full">
                    <BarChart2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <div className="flex-1">
                      <div className="font-medium text-sm">Balanced</div>
                      <div className="text-xs text-muted-foreground">Clear with moderate details</div>
                    </div>
                  </div>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => onComplexityChange("detailed")}
                  className={`cursor-pointer ${complexity === "detailed" ? "bg-accent" : ""}`}
                >
                  <div className="flex items-start gap-3 w-full">
                    <Layers className="h-4 w-4 text-purple-500 mt-0.5 shrink-0" />
                    <div className="flex-1">
                      <div className="font-medium text-sm">Detailed</div>
                      <div className="text-xs text-muted-foreground">Comprehensive explanations</div>
                    </div>
                  </div>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-1 sm:gap-1 sm:gap-2">
        {/* Dataset Builder Link */}
        <Link href="/dataset">
          <Button
            variant="outline"
            size="sm"
            className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm h-8 sm:h-9 px-2 sm:px-3 active:scale-95 touch-manipulation"
          >
            <Database className="h-3 w-3 sm:h-4 sm:w-4" />
            <span className="hidden lg:inline">Dataset Builder</span>
          </Button>
        </Link>

        {/* Training Link */}
        <Link href="/training">
          <Button
            variant="outline"
            size="sm"
            className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm h-8 sm:h-9 px-2 sm:px-3 active:scale-95 touch-manipulation"
          >
            <GraduationCap className="h-3 w-3 sm:h-4 sm:w-4" />
            <span className="hidden lg:inline">Training</span>
          </Button>
        </Link>

        {/* Model Selector */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="outline"
              className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm h-8 sm:h-9 px-2 sm:px-3 active:scale-95 touch-manipulation"
              disabled={modelsLoading}
            >
              {modelsLoading ? (
                <>
                  <Loader2 className="h-3 w-3 sm:h-4 sm:w-4 animate-spin" />
                  <span className="hidden md:inline">Loading...</span>
                </>
              ) : (
                <>
                  <Cpu className="h-3 w-3 sm:h-4 sm:w-4" />
                  <span className="hidden md:inline max-w-[80px] lg:max-w-[120px] truncate">
                    {selectedModel || "Select Model"}
                  </span>
                  <ChevronDown className="h-3 w-3 sm:h-4 sm:w-4" />
                </>
              )}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="center" className="w-48 sm:w-56 max-h-64 overflow-y-auto">
            {availableModels.length === 0 ? (
              <DropdownMenuItem disabled>
                <span className="text-muted-foreground">
                  No models found. Is Ollama running?
                </span>
              </DropdownMenuItem>
            ) : (
              availableModels.map((model) => (
                <DropdownMenuItem
                  key={model.name}
                  onClick={() => onModelChange(model.name)}
                  className={selectedModel === model.name ? "bg-accent" : ""}
                >
                  <Cpu className="mr-2 h-4 w-4" />
                  <span className="truncate">{model.name}</span>
                </DropdownMenuItem>
              ))
            )}
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Theme Toggle */}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          className="text-muted-foreground hover:text-foreground active:scale-95 touch-manipulation h-8 w-8 sm:h-10 sm:w-10"
        >
          {mounted && theme === "dark" ? (
            <Sun className="h-4 w-4 sm:h-5 sm:w-5" />
          ) : (
            <Moon className="h-4 w-4 sm:h-5 sm:w-5" />
          )}
          <span className="sr-only">Toggle theme</span>
        </Button>

        {/* Settings Dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="text-muted-foreground hover:text-foreground active:scale-95 touch-manipulation h-8 w-8 sm:h-10 sm:w-10"
            >
              <Settings className="h-4 w-4 sm:h-5 sm:w-5" />
              <span className="sr-only">Settings</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuItem>
              <Settings className="mr-2 h-4 w-4" />
              Settings
            </DropdownMenuItem>
            <DropdownMenuItem>
              <HelpCircle className="mr-2 h-4 w-4" />
              Help & FAQ
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              {mounted && theme === "dark" ? (
                <>
                  <Sun className="mr-2 h-4 w-4" />
                  Light Mode
                </>
              ) : (
                <>
                  <Moon className="mr-2 h-4 w-4" />
                  Dark Mode
                </>
              )}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  )
}
