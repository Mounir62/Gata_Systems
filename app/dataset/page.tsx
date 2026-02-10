"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { ThemeProvider } from "@/components/theme-provider"
import { useTheme } from "@/components/theme-provider"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Download, Loader2, FileText, ChevronDown, Cpu, Moon, Sun, ArrowLeft } from "lucide-react"
import Link from "next/link"

interface OllamaModel {
  name: string
  modified_at: string
  size: number
}

interface QAPair {
  question: string
  answer: string
}

function DatasetBuilderContent() {
  const { theme, setTheme } = useTheme()
  const [questions, setQuestions] = React.useState("")
  const [selectedModel, setSelectedModel] = React.useState("")
  const [availableModels, setAvailableModels] = React.useState<OllamaModel[]>([])
  const [modelsLoading, setModelsLoading] = React.useState(true)
  const [isGenerating, setIsGenerating] = React.useState(false)
  const [results, setResults] = React.useState<QAPair[]>([])
  const [error, setError] = React.useState<string | null>(null)
  const [mounted, setMounted] = React.useState(false)
  const [progress, setProgress] = React.useState(0)
  const [progressText, setProgressText] = React.useState("")
  const [documents, setDocuments] = React.useState<string[]>([])
  const [selectedDocument, setSelectedDocument] = React.useState<string | null>(null)

  React.useEffect(() => {
    setMounted(true)
  }, [])

  // Fetch available documents on mount
  React.useEffect(() => {
    const fetchDocuments = async () => {
      try {
        const res = await fetch("/api/documents/list")
        if (res.ok) {
          const data = await res.json()
          setDocuments(data.documents || [])
        }
      } catch {
        // ignore
      }
    }
    fetchDocuments()
  }, [])

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

  const handleGenerate = async () => {
    setError(null)
    const questionList = questions
      .split("\n")
      .map((q) => q.trim())
      .filter((q) => q.length > 0)

    if (questionList.length === 0) {
      setError("Please enter at least one question.")
      return
    }

    if (!selectedModel) {
      setError("Please select a model.")
      return
    }

    setIsGenerating(true)
    setResults([])
    setProgress(0)
    setProgressText("Starting...")

    try {
      const response = await fetch("/api/dataset/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          questions: questionList,
          model: selectedModel,
          selectedDocument,
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to generate dataset")
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      const newResults: QAPair[] = []

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value)
          const lines = chunk.split("\n")

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = line.slice(6)
              try {
                const json = JSON.parse(data)
                if (json.type === "progress") {
                  const { current, total, result } = json
                  newResults.push(result)
                  setResults([...newResults])
                  setProgress((current / total) * 100)
                  setProgressText(`Processing ${current}/${total}...`)
                } else if (json.type === "done") {
                  setProgressText("Complete!")
                }
              } catch {
                // Skip invalid JSON
              }
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate dataset")
    } finally {
      setIsGenerating(false)
    }
  }

  const downloadCSV = () => {
    if (results.length === 0) return

    // Create CSV content
    const csvRows = [
      ["Question", "Answer"], // Header
      ...results.map((pair) => [
        `"${pair.question.replace(/"/g, '""')}"`,
        `"${pair.answer.replace(/"/g, '""')}"`,
      ]),
    ]

    const csvContent = csvRows.map((row) => row.join(",")).join("\n")
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = `sft_dataset_${Date.now()}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="flex h-14 sm:h-16 items-center justify-between border-b border-border bg-background px-3 sm:px-4">
        <div className="flex items-center gap-2 sm:gap-3">
          <Link href="/">
            <Button variant="ghost" size="icon" className="active:scale-95 touch-manipulation">
              <ArrowLeft className="h-4 w-4 sm:h-5 sm:w-5" />
            </Button>
          </Link>
          <div className="flex items-center gap-2">
            <img src="/images/image.png" alt="GATA Logo" className="h-6 sm:h-7 w-auto" />
            <span className="text-xs sm:text-sm font-medium text-muted-foreground hidden sm:inline">
              Dataset Builder
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
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
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto max-w-5xl p-3 sm:p-4 md:p-6">
        <div className="mb-4 sm:mb-6">
          <h1 className="text-2xl sm:text-3xl font-bold text-foreground mb-2">
            SFT Training Dataset Builder
          </h1>
          <p className="text-sm sm:text-base text-muted-foreground">
            Create supervised fine-tuning datasets by generating question-answer pairs using your selected model.
          </p>
        </div>

        <div className="grid gap-4 sm:gap-6 lg:grid-cols-2">
          {/* Input Section */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg sm:text-xl">Input Questions</CardTitle>
              <CardDescription className="text-sm">
                Enter your questions (one per line)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                placeholder="What is machine learning?&#10;Explain neural networks.&#10;How does backpropagation work?"
                value={questions}
                onChange={(e) => setQuestions(e.target.value)}
                className="min-h-[200px] sm:min-h-[300px] font-mono text-xs sm:text-sm touch-manipulation"
              />

              {/* Document Selector (Optional) */}
              {documents.length > 0 && (
                <div className="space-y-2 sm:space-y-3">
                  <label className="text-sm font-medium">Document (Optional)</label>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="outline"
                        className="w-full justify-between text-xs sm:text-sm h-9 sm:h-10 active:scale-95 touch-manipulation"
                      >
                        <div className="flex items-center gap-2 truncate">
                          <FileText className="h-3 w-3 sm:h-4 sm:w-4 shrink-0" />
                          <span className="truncate">
                            {selectedDocument || "No document (general QA)"}
                          </span>
                        </div>
                        <ChevronDown className="h-3 w-3 sm:h-4 sm:w-4 ml-2 shrink-0" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent className="w-full max-h-64 overflow-y-auto">
                      <DropdownMenuItem
                        onClick={() => setSelectedDocument(null)}
                        className={!selectedDocument ? "bg-accent" : ""}
                      >
                        <span className="text-xs sm:text-sm">No document (general QA)</span>
                      </DropdownMenuItem>
                      {documents.map((doc) => (
                        <DropdownMenuItem
                          key={doc}
                          onClick={() => setSelectedDocument(doc)}
                          className={selectedDocument === doc ? "bg-accent" : ""}
                        >
                          <FileText className="mr-2 h-3 w-3 sm:h-4 sm:w-4" />
                          <span className="truncate text-xs sm:text-sm">{doc}</span>
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              )}

              <div className="space-y-2 sm:space-y-3">
                <label className="text-sm font-medium">Select Model</label>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      className="w-full justify-between text-xs sm:text-sm h-9 sm:h-10 active:scale-95 touch-manipulation"
                      disabled={modelsLoading}
                    >
                      {modelsLoading ? (
                        <>
                          <Loader2 className="h-3 w-3 sm:h-4 sm:w-4 animate-spin mr-2" />
                          <span className="text-xs sm:text-sm">Loading models...</span>
                        </>
                      ) : (
                        <>
                          <div className="flex items-center gap-2 truncate">
                            <Cpu className="h-3 w-3 sm:h-4 sm:w-4 shrink-0" />
                            <span className="truncate">{selectedModel || "Select Model"}</span>
                          </div>
                          <ChevronDown className="h-3 w-3 sm:h-4 sm:w-4 ml-2 shrink-0" />
                        </>
                      )}
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="w-full max-h-64 overflow-y-auto">
                    {availableModels.length === 0 ? (
                      <DropdownMenuItem disabled>
                        <span className="text-xs sm:text-sm text-muted-foreground">
                          No models found
                        </span>
                      </DropdownMenuItem>
                    ) : (
                      availableModels.map((model) => (
                        <DropdownMenuItem
                          key={model.name}
                          onClick={() => setSelectedModel(model.name)}
                          className={selectedModel === model.name ? "bg-accent" : ""}
                        >
                          <Cpu className="mr-2 h-3 w-3 sm:h-4 sm:w-4" />
                          <span className="truncate text-xs sm:text-sm">{model.name}</span>
                        </DropdownMenuItem>
                      ))
                    )}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              {isGenerating && (
                <div className="space-y-2">
                  <Progress value={progress} className="w-full" />
                  <p className="text-xs text-center text-muted-foreground">
                    {progressText}
                  </p>
                </div>
              )}

              <Button
                onClick={handleGenerate}
                disabled={isGenerating || !questions.trim() || !selectedModel}
                className="w-full h-9 sm:h-10 text-sm active:scale-95 touch-manipulation"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="mr-2 h-3 w-3 sm:h-4 sm:w-4 animate-spin" />
                    <span className="text-xs sm:text-sm">Generating Dataset...</span>
                  </>
                ) : (
                  <>
                    <FileText className="mr-2 h-3 w-3 sm:h-4 sm:w-4" />
                    <span className="text-xs sm:text-sm">Generate Dataset</span>
                  </>
                )}
              </Button>

              {error && (
                <p className="text-sm text-destructive">{error}</p>
              )}
            </CardContent>
          </Card>

          {/* Results Section */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg sm:text-xl">Results</CardTitle>
              <CardDescription className="text-sm">
                {results.length > 0
                  ? `${results.length} question-answer pairs generated`
                  : "Generated Q&A pairs will appear here"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {results.length > 0 ? (
                <div className="space-y-4">
                  <div className="max-h-[200px] sm:max-h-[300px] overflow-y-auto space-y-2 sm:space-y-3 rounded-lg border border-border p-2 sm:p-3">
                    {results.map((pair, index) => (
                      <div
                        key={index}
                        className="rounded-md border border-border bg-card p-2 sm:p-3 space-y-2"
                      >
                        <div>
                          <span className="text-xs font-semibold text-primary">Q:</span>
                          <p className="text-xs sm:text-sm mt-1">{pair.question}</p>
                        </div>
                        <div>
                          <span className="text-xs font-semibold text-secondary">A:</span>
                          <p className="text-xs sm:text-sm mt-1 text-muted-foreground">
                            {pair.answer.slice(0, 150)}
                            {pair.answer.length > 150 ? "..." : ""}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>

                  <Button onClick={downloadCSV} className="w-full h-9 sm:h-10 text-sm active:scale-95 touch-manipulation">
                    <Download className="mr-2 h-3 w-3 sm:h-4 sm:w-4" />
                    <span className="text-xs sm:text-sm">Download CSV</span>
                  </Button>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-8 sm:py-12 text-center px-4">
                  <FileText className="h-10 w-10 sm:h-12 sm:w-12 text-muted-foreground/50 mb-3" />
                  <p className="text-xs sm:text-sm text-muted-foreground">
                    No results yet. Enter questions and generate dataset.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}

export default function DatasetBuilderPage() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
      <DatasetBuilderContent />
    </ThemeProvider>
  )
}
