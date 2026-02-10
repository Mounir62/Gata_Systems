"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ThemeProvider, useTheme } from "@/components/theme-provider"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  ArrowLeft,
  Moon,
  Sun,
  Upload,
  Play,
  Square,
  Save,
  Cpu,
  Loader2,
  FileText,
  ChevronDown,
  Activity,
  BarChart3,
  MessageSquare,
  Send,
  Bot,
  User,
  CheckCircle2,
  AlertCircle,
  Power,
  PowerOff,
  Server,
  Download,
  Circle,
} from "lucide-react"
import Link from "next/link"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts"

// ── Types ----------------------------------------------------------------- //

interface DeviceInfo {
  device: string
  name: string
  memory_gb: number
  error?: string
}

interface TrainingMetrics {
  step: number
  total_steps: number
  epoch: number
  loss: number | null
  learning_rate: number | null
  elapsed_seconds: number
  system: {
    cpu_percent: number
    ram_percent: number
    gpu_mem_used_gb?: number
    gpu_mem_total_gb?: number
    gpu_util?: number
  }
}

interface LogEntry {
  timestamp: string
  message: string
}

interface TestMessage {
  id: string
  role: "user" | "assistant"
  content: string
}

interface OllamaModel {
  name: string
  modified_at: string
  size: number
}

type StepStatus = "not-started" | "in-progress" | "completed" | "error"

interface PipelineStep {
  id: string
  label: string
  status: StepStatus
  detail: string
}

interface DownloadProgress {
  filename: string
  percent: number
  speedBytesPerSec: number
  downloadedBytes: number
  totalBytes: number
}

const INITIAL_PIPELINE: PipelineStep[] = [
  { id: "detect_device", label: "Detect Device", status: "not-started", detail: "" },
  { id: "load_dataset", label: "Load Dataset", status: "not-started", detail: "" },
  { id: "load_libraries", label: "Load Libraries", status: "not-started", detail: "" },
  { id: "prepare_dataset", label: "Prepare Dataset", status: "not-started", detail: "" },
  { id: "download_model", label: "Download Model", status: "not-started", detail: "" },
  { id: "apply_lora", label: "Apply LoRA", status: "not-started", detail: "" },
  { id: "training", label: "Training", status: "not-started", detail: "" },
  { id: "saving", label: "Save Model", status: "not-started", detail: "" },
  { id: "convert_gguf", label: "Convert to GGUF", status: "not-started", detail: "" },
]

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B"
  const k = 1024
  const sizes = ["B", "KB", "MB", "GB"]
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i]
}

// ── Defaults --------------------------------------------------------------- //

interface AvailableDevice {
  device: string
  device_index: number
  name: string
  memory_gb: number
}

const DEFAULT_PARAMS = {
  baseModel: "Qwen/Qwen3-0.6B",
  epochs: 3,
  batchSize: 4,
  learningRate: 2e-4,
  maxSeqLength: 2048,
  loraR: 16,
  loraAlpha: 32,
  gradientAccumulationSteps: 4,
  warmupRatio: 0.03,
  weightDecay: 0.01,
  saveSteps: 50,
}

/** Return safe training defaults based on device and available memory. */
function getDeviceDefaults(device: string, memoryGb: number): Partial<typeof DEFAULT_PARAMS> {
  if (device === "cuda") {
    if (memoryGb >= 16) {
      return { batchSize: 4, gradientAccumulationSteps: 4, maxSeqLength: 2048, loraR: 16, loraAlpha: 32 }
    }
    if (memoryGb >= 8) {
      return { batchSize: 2, gradientAccumulationSteps: 8, maxSeqLength: 1024, loraR: 8, loraAlpha: 16 }
    }
    // < 8 GB VRAM
    return { batchSize: 1, gradientAccumulationSteps: 16, maxSeqLength: 512, loraR: 8, loraAlpha: 16 }
  }

  if (device === "mps") {
    // Apple Silicon — shares system RAM, conservative to avoid swapping
    if (memoryGb >= 32) {
      return { batchSize: 2, gradientAccumulationSteps: 8, maxSeqLength: 1024, loraR: 8, loraAlpha: 16 }
    }
    if (memoryGb >= 16) {
      return { batchSize: 1, gradientAccumulationSteps: 16, maxSeqLength: 512, loraR: 8, loraAlpha: 16 }
    }
    // < 16 GB
    return { batchSize: 1, gradientAccumulationSteps: 16, maxSeqLength: 256, loraR: 4, loraAlpha: 8 }
  }

  // CPU — most conservative
  return { batchSize: 1, gradientAccumulationSteps: 16, maxSeqLength: 512, loraR: 8, loraAlpha: 16 }
}

// ── Main Content ----------------------------------------------------------- //

function TrainingContent() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = React.useState(false)

  // Config
  const [params, setParams] = React.useState(DEFAULT_PARAMS)
  const [csvFile, setCsvFile] = React.useState<File | null>(null)
  const [csvPreview, setCsvPreview] = React.useState<string[]>([])
  const [hfToken, setHfToken] = React.useState("") // Optional HF token for gated models
  const fileInputRef = React.useRef<HTMLInputElement>(null)

  // Device
  const [deviceInfo, setDeviceInfo] = React.useState<DeviceInfo | null>(null)
  const [deviceLoading, setDeviceLoading] = React.useState(true)
  const [availableDevices, setAvailableDevices] = React.useState<AvailableDevice[]>([])
  const [selectedDevice, setSelectedDevice] = React.useState<string | null>(null)
  const [paramsAdjusted, setParamsAdjusted] = React.useState(false)

  // Training state
  const [isTraining, setIsTraining] = React.useState(false)
  const [trainingDone, setTrainingDone] = React.useState(false)
  const [outputDir, setOutputDir] = React.useState<string | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [statusMessage, setStatusMessage] = React.useState("")
  const [metricsHistory, setMetricsHistory] = React.useState<TrainingMetrics[]>([])
  const [currentStep, setCurrentStep] = React.useState(0)
  const [totalSteps, setTotalSteps] = React.useState(0)
  const [logs, setLogs] = React.useState<LogEntry[]>([])

  // Save to Ollama
  const [ollamaModelName, setOllamaModelName] = React.useState("")
  const [saving, setSaving] = React.useState(false)
  const [saveResult, setSaveResult] = React.useState<string | null>(null)

  // Test model
  const [testMessages, setTestMessages] = React.useState<TestMessage[]>([])
  const [testInput, setTestInput] = React.useState("")
  const [testLoading, setTestLoading] = React.useState(false)
  const [testModel, setTestModel] = React.useState("")
  const [availableModels, setAvailableModels] = React.useState<OllamaModel[]>([])

  // Documents for RAG
  const [documents, setDocuments] = React.useState<string[]>([])
  const [testDocument, setTestDocument] = React.useState<string | null>(null)

  // Controlled tabs
  const [activeTab, setActiveTab] = React.useState("config")

  // Pipeline steps
  const [pipelineSteps, setPipelineSteps] = React.useState<PipelineStep[]>(INITIAL_PIPELINE)
  const [downloadProgress, setDownloadProgress] = React.useState<DownloadProgress | null>(null)

  // Python Training Server
  const [serverRunning, setServerRunning] = React.useState(false)
  const [serverLoading, setServerLoading] = React.useState(false)
  const [serverLogs, setServerLogs] = React.useState<string[]>([])
  const [serverMessage, setServerMessage] = React.useState<string | null>(null)

  const logsEndRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => { setMounted(true) }, [])

  // Detect all available devices on mount
  React.useEffect(() => {
    Promise.all([
      fetch("/api/training/device").then(r => r.json()).catch(() => ({ device: "cpu", name: "CPU", memory_gb: 0 })),
      fetch("/api/training/devices").then(r => r.json()).catch(() => ({ devices: [] })),
    ]).then(([bestDevice, allDevices]) => {
      setDeviceInfo(bestDevice)
      const devices: AvailableDevice[] = allDevices.devices?.length
        ? allDevices.devices
        : [{ device: bestDevice.device, device_index: 0, name: bestDevice.name, memory_gb: bestDevice.memory_gb }]
      setAvailableDevices(devices)

      // Auto-select best available device
      const best = bestDevice.device as string
      setSelectedDevice(best)

      // Auto-adjust defaults for the detected device
      const match = devices.find((d: AvailableDevice) => d.device === best)
      const mem = match?.memory_gb ?? bestDevice.memory_gb ?? 0
      const adjusted = getDeviceDefaults(best, mem)
      setParams(prev => ({ ...prev, ...adjusted }))
      setParamsAdjusted(true)
    }).finally(() => setDeviceLoading(false))
  }, [])

  // Fetch models + documents
  React.useEffect(() => {
    fetch("/api/ollama/models").then(r => r.json()).then(d => setAvailableModels(d.models || [])).catch(() => {})
    fetch("/api/documents/list").then(r => r.json()).then(d => setDocuments(d.documents || [])).catch(() => {})
  }, [saveResult])

  // Auto-scroll logs
  React.useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  // Poll Python training server status
  React.useEffect(() => {
    const checkServer = () => {
      fetch("/api/training/server")
        .then((r) => r.json())
        .then((d) => {
          setServerRunning(d.running)
          if (d.logs) setServerLogs(d.logs)
        })
        .catch(() => setServerRunning(false))
    }
    checkServer()
    const interval = setInterval(checkServer, 5000)
    return () => clearInterval(interval)
  }, [])

  // ── Start / stop Python server ----------------------------------------- //
  const startServer = async () => {
    setServerLoading(true)
    setServerMessage(null)
    try {
      const res = await fetch("/api/training/server", { method: "POST" })
      const data = await res.json()
      setServerRunning(data.running)
      setServerMessage(data.message || data.error || null)
      if (data.logs) setServerLogs(data.logs)
    } catch {
      setServerMessage("Failed to start server.")
    } finally {
      setServerLoading(false)
    }
  }

  const stopServer = async () => {
    setServerLoading(true)
    setServerMessage(null)
    try {
      const res = await fetch("/api/training/server", { method: "DELETE" })
      const data = await res.json()
      setServerRunning(data.running)
      setServerMessage(data.message || null)
    } catch {
      setServerMessage("Failed to stop server.")
    } finally {
      setServerLoading(false)
    }
  }

  // ── Device selection --------------------------------------------------- //
  const handleDeviceChange = (deviceType: string) => {
    setSelectedDevice(deviceType)
    const dev = availableDevices.find(d => d.device === deviceType)
    if (dev) {
      setDeviceInfo({ device: dev.device, name: dev.name, memory_gb: dev.memory_gb })
      const adjusted = getDeviceDefaults(dev.device, dev.memory_gb)
      setParams(prev => ({ ...prev, ...adjusted }))
      setParamsAdjusted(true)
    }
  }

  // ── CSV upload --------------------------------------------------------- //
  const handleCsvUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setCsvFile(file)
    const reader = new FileReader()
    reader.onload = (ev) => {
      const text = ev.target?.result as string
      const lines = text.split("\n").filter(l => l.trim())
      setCsvPreview(lines.slice(0, 4))
    }
    reader.readAsText(file)
  }

  // ── Update param helper ------------------------------------------------ //
  const setParam = <K extends keyof typeof DEFAULT_PARAMS>(key: K, value: typeof DEFAULT_PARAMS[K]) =>
    setParams(prev => ({ ...prev, [key]: value }))

  // ── Start training ----------------------------------------------------- //
  const startTraining = async () => {
    if (!csvFile) { setError("Please upload a CSV dataset."); return }
    if (!params.baseModel.trim()) { setError("Please specify a base model."); return }

    setError(null)
    setIsTraining(true)
    setTrainingDone(false)
    setMetricsHistory([])
    setCurrentStep(0)
    setTotalSteps(0)
    setLogs([])
    setStatusMessage("Starting training…")
    setOutputDir(null)
    setSaveResult(null)
    setPipelineSteps(INITIAL_PIPELINE)
    setDownloadProgress(null)
    setActiveTab("monitor")

    const formData = new FormData()
    formData.append("file", csvFile)
    formData.append("params", JSON.stringify({ ...params, trainingDevice: selectedDevice, hfToken: hfToken.trim() || undefined }))

    try {
      const response = await fetch("/api/training/start", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.error || "Failed to start training")
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (reader) {
        let buf = ""
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buf += decoder.decode(value, { stream: true })
          const lines = buf.split("\n")
          buf = lines.pop() || ""

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue
            const raw = line.slice(6).trim()
            if (!raw) continue

            try {
              const msg = JSON.parse(raw)
              const now = new Date().toLocaleTimeString()

              switch (msg.type) {
                case "status":
                  setStatusMessage(msg.message)
                  setLogs(prev => [...prev, { timestamp: now, message: msg.message }])
                  break

                case "device":
                  setDeviceInfo(msg)
                  break

                case "training_start":
                  setTotalSteps(msg.total_steps)
                  setLogs(prev => [...prev, {
                    timestamp: now,
                    message: `Training started: ${msg.total_examples} examples, ${msg.epochs} epochs, ${msg.total_steps} steps`,
                  }])
                  break

                case "metrics":
                  setCurrentStep(msg.step)
                  setTotalSteps(msg.total_steps)
                  setMetricsHistory(prev => [...prev, msg])
                  setStatusMessage(`Step ${msg.step}/${msg.total_steps} | Epoch ${msg.epoch} | Loss: ${msg.loss?.toFixed(4) ?? "–"}`)
                  break

                case "pipeline_step":
                  setPipelineSteps(prev => prev.map(s =>
                    s.id === msg.step
                      ? { ...s, status: msg.status as StepStatus, detail: msg.detail || s.detail }
                      : s
                  ))
                  break

                case "download_progress":
                  if (msg.percent < 100) {
                    setDownloadProgress({
                      filename: msg.filename,
                      percent: msg.percent,
                      speedBytesPerSec: msg.speed_bytes_per_sec,
                      downloadedBytes: msg.downloaded_bytes,
                      totalBytes: msg.total_bytes,
                    })
                  } else {
                    setDownloadProgress(null)
                  }
                  break

                case "log":
                  setLogs(prev => [...prev, { timestamp: now, message: msg.message }])
                  break

                case "done":
                  setOutputDir(msg.output_dir)
                  setTrainingDone(true)
                  setStatusMessage("Training complete!")
                  setLogs(prev => [...prev, { timestamp: now, message: "✅ Training complete!" }])
                  break

                case "error":
                  setError(msg.message)
                  setLogs(prev => [...prev, { timestamp: now, message: `❌ ${msg.message}` }])
                  break
              }
            } catch {
              // skip invalid JSON
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed")
    } finally {
      setIsTraining(false)
    }
  }

  // ── Stop training ------------------------------------------------------ //
  const stopTraining = async () => {
    try {
      await fetch("/api/training/start", { method: "DELETE" })
      setIsTraining(false)
      setStatusMessage("Training cancelled.")
    } catch {
      // ignore
    }
  }

  // ── Save to Ollama ----------------------------------------------------- //
  const saveToOllama = async () => {
    if (!outputDir || !ollamaModelName.trim()) return
    setSaving(true)
    setSaveResult(null)
    try {
      const res = await fetch("/api/training/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ outputDir, modelName: ollamaModelName.trim() }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error)
      setSaveResult(data.message)
      setTestModel(ollamaModelName.trim())
    } catch (err) {
      setSaveResult(`Error: ${err instanceof Error ? err.message : "Save failed"}`)
    } finally {
      setSaving(false)
    }
  }

  // ── Test model --------------------------------------------------------- //
  const sendTestMessage = async () => {
    if (!testInput.trim() || !testModel) return
    const userMsg: TestMessage = { id: `u-${Date.now()}`, role: "user", content: testInput.trim() }
    setTestMessages(prev => [...prev, userMsg])
    setTestInput("")
    setTestLoading(true)

    try {
      const res = await fetch("/api/ollama", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: testModel,
          messages: [...testMessages, userMsg].map(m => ({ role: m.role, content: m.content })),
          selectedDocument: testDocument,
        }),
      })

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()
      let fullContent = ""

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          const chunk = decoder.decode(value)
          for (const line of chunk.split("\n")) {
            if (line.startsWith("data: ")) {
              const data = line.slice(6)
              if (data === "[DONE]") continue
              try {
                const json = JSON.parse(data)
                if (json.content) fullContent += json.content
              } catch { /* skip */ }
            }
          }
        }
      }

      setTestMessages(prev => [...prev, {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: fullContent || "No response.",
      }])
    } catch {
      setTestMessages(prev => [...prev, {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: "Error: Could not reach the model.",
      }])
    } finally {
      setTestLoading(false)
    }
  }

  // ── Derived data for charts -------------------------------------------- //
  const lossData = metricsHistory
    .filter(m => m.loss != null)
    .map(m => ({ step: m.step, loss: Number(m.loss!.toFixed(4)), lr: m.learning_rate }))

  const systemData = metricsHistory.map(m => ({
    step: m.step,
    cpu: m.system.cpu_percent,
    ram: m.system.ram_percent,
    gpuMem: m.system.gpu_mem_used_gb ?? 0,
  }))

  const progressPercent = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0

  // ── Render ------------------------------------------------------------- //
  return (
    <div className="min-h-screen bg-background">
      {/* Header — same pattern as dataset page */}
      <header className="flex h-14 sm:h-16 items-center justify-between border-b border-border bg-background px-3 sm:px-4">
        <div className="flex items-center gap-2 sm:gap-3">
          <Link href="/">
            <Button variant="ghost" size="icon" className="active:scale-95 touch-manipulation"><ArrowLeft className="h-4 w-4 sm:h-5 sm:w-5" /></Button>
          </Link>
          <div className="flex items-center gap-2">
            <img src="/images/image.png" alt="GATA Logo" className="h-6 sm:h-7 w-auto" />
            <span className="text-xs sm:text-sm font-medium text-muted-foreground hidden sm:inline">Model Training</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} className="text-muted-foreground hover:text-foreground active:scale-95 touch-manipulation h-8 w-8 sm:h-10 sm:w-10">
            {mounted && theme === "dark" ? <Sun className="h-4 w-4 sm:h-5 sm:w-5" /> : <Moon className="h-4 w-4 sm:h-5 sm:w-5" />}
          </Button>
        </div>
      </header>

      <main className="container mx-auto max-w-6xl p-3 sm:p-4 md:p-6">
        <div className="mb-4 sm:mb-6">
          <h1 className="text-2xl sm:text-3xl font-bold text-foreground mb-2">LLM Model Training</h1>
          <p className="text-sm sm:text-base text-muted-foreground">
            Fine-tune a Hugging Face model with LoRA using your SFT dataset, then save it to Ollama.
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4 sm:space-y-6">
          <TabsList className="grid w-full grid-cols-4 h-auto">
            <TabsTrigger value="config" className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm py-2 sm:py-2.5"><Cpu className="h-3 w-3 sm:h-4 sm:w-4" /><span className="hidden sm:inline">Configure</span></TabsTrigger>
            <TabsTrigger value="monitor" className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm py-2 sm:py-2.5"><Activity className="h-3 w-3 sm:h-4 sm:w-4" /><span className="hidden sm:inline">Monitor</span></TabsTrigger>
            <TabsTrigger value="charts" className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm py-2 sm:py-2.5"><BarChart3 className="h-3 w-3 sm:h-4 sm:w-4" /><span className="hidden sm:inline">Charts</span></TabsTrigger>
            <TabsTrigger value="test" className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm py-2 sm:py-2.5"><MessageSquare className="h-3 w-3 sm:h-4 sm:w-4" /><span className="hidden sm:inline">Test</span></TabsTrigger>
          </TabsList>

          {/* ═══════════════ TAB 1: CONFIGURE ═══════════════ */}
          <TabsContent value="config">
            {/* Server Status Card */}
            <Card className="mb-4 sm:mb-6">
              <CardContent className="flex flex-col sm:flex-row flex-wrap items-start sm:items-center justify-between gap-3 sm:gap-4 py-3 sm:py-4">
                <div className="flex items-center gap-2 sm:gap-3">
                  <Server className="h-4 w-4 sm:h-5 sm:w-5 text-muted-foreground shrink-0" />
                  <div>
                    <p className="text-sm font-medium">Python Training Server</p>
                    <p className="text-xs text-muted-foreground">
                      Required for model training. Runs on port 8000.
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 sm:gap-3 w-full sm:w-auto">
                  <Badge
                    variant={serverRunning ? "default" : "secondary"}
                    className={`gap-1.5 text-xs ${serverRunning ? "bg-green-600 hover:bg-green-700 text-white" : ""}`}
                  >
                    <span className={`h-2 w-2 rounded-full ${serverRunning ? "bg-green-300 animate-pulse" : "bg-muted-foreground"}`} />
                    {serverRunning ? "Running" : "Stopped"}
                  </Badge>

                  {serverRunning ? (
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={stopServer}
                      disabled={serverLoading}
                      className="gap-1.5 h-8 sm:h-9 text-xs sm:text-sm active:scale-95 touch-manipulation flex-1 sm:flex-initial"
                    >
                      {serverLoading ? (
                        <Loader2 className="h-3 w-3 sm:h-3.5 sm:w-3.5 animate-spin" />
                      ) : (
                        <PowerOff className="h-3 w-3 sm:h-3.5 sm:w-3.5" />
                      )}
                      <span className="hidden sm:inline">Stop Server</span>
                      <span className="sm:hidden">Stop</span>
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      onClick={startServer}
                      disabled={serverLoading}
                      className="gap-1.5 h-8 sm:h-9 text-xs sm:text-sm active:scale-95 touch-manipulation flex-1 sm:flex-initial"
                    >
                      {serverLoading ? (
                        <Loader2 className="h-3 w-3 sm:h-3.5 sm:w-3.5 animate-spin" />
                      ) : (
                        <Power className="h-3 w-3 sm:h-3.5 sm:w-3.5" />
                      )}
                      <span className="hidden sm:inline">Start Server</span>
                      <span className="sm:hidden">Start</span>
                    </Button>
                  )}
                </div>

                {serverMessage && (
                  <p className={`w-full text-xs ${serverMessage.toLowerCase().includes("error") || serverMessage.toLowerCase().includes("fail") ? "text-destructive" : "text-muted-foreground"}`}>
                    {serverMessage}
                  </p>
                )}
              </CardContent>
            </Card>

            <div className="grid gap-4 sm:gap-6 lg:grid-cols-2">
              {/* Left: Model & Dataset */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg sm:text-xl">Model & Dataset</CardTitle>
                  <CardDescription className="text-sm">Select base model and upload training data</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 sm:space-y-4">
                  <div className="space-y-2">
                    <Label className="text-sm">Hugging Face Model Name</Label>
                    <Input
                      placeholder="e.g. unsloth/Llama-3.2-1B"
                      value={params.baseModel}
                      onChange={(e) => setParam("baseModel", e.target.value)}
                      className="h-9 sm:h-10 text-sm touch-manipulation"
                    />
                    <p className="text-xs text-muted-foreground">
                      Paste the full HF model identifier
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label>Hugging Face Token (Optional)</Label>
                    <Input
                      type="password"
                      placeholder="hf_..."
                      value={hfToken}
                      onChange={(e) => setHfToken(e.target.value)}
                    />
                    <p className="text-xs text-muted-foreground">
                      Required for gated models (Llama, Gemma, etc.). Leave empty for public models.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-sm">Training Dataset (CSV)</Label>
                    <div className="flex items-center gap-2">
                      <input ref={fileInputRef} type="file" accept=".csv" onChange={handleCsvUpload} className="hidden" />
                      <Button variant="outline" className="w-full justify-start gap-2 h-9 sm:h-10 text-sm active:scale-95 touch-manipulation" onClick={() => fileInputRef.current?.click()}>
                        <Upload className="h-3 w-3 sm:h-4 sm:w-4" />
                        <span className="truncate">{csvFile ? csvFile.name : "Upload CSV file"}</span>
                      </Button>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Same format as Dataset Builder output (Question, Answer columns)
                    </p>
                  </div>

                  {csvPreview.length > 0 && (
                    <div className="rounded-md border border-border bg-muted/50 p-3">
                      <p className="text-xs font-medium mb-1 text-muted-foreground">Preview (first 3 rows):</p>
                      {csvPreview.map((line, i) => (
                        <p key={i} className="text-xs font-mono truncate text-muted-foreground">{line}</p>
                      ))}
                    </div>
                  )}

                  {/* Device Selector */}
                  <div className="rounded-md border border-border p-2 sm:p-3 space-y-2">
                    <div className="flex items-center gap-2 mb-1">
                      <Cpu className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground" />
                      <span className="text-xs sm:text-sm font-medium">Training Device</span>
                    </div>
                    {deviceLoading ? (
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Loader2 className="h-3 w-3 animate-spin" /> Detecting…
                      </div>
                    ) : (
                      <>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="outline" className="w-full justify-between text-xs sm:text-sm h-8 sm:h-9 active:scale-95 touch-manipulation" disabled={isTraining}>
                              <div className="flex items-center gap-1 sm:gap-2 truncate">
                                <Badge variant={selectedDevice === "cuda" ? "default" : selectedDevice === "mps" ? "secondary" : "outline"}
                                  className="text-[10px] px-1 sm:px-1.5 py-0">
                                  {(selectedDevice || "cpu").toUpperCase()}
                                </Badge>
                                <span className="truncate text-xs sm:text-sm">
                                  {availableDevices.find(d => d.device === selectedDevice)?.name || "Select device"}
                                </span>
                              </div>
                              <ChevronDown className="h-3 w-3 sm:h-3.5 sm:w-3.5 ml-2 shrink-0" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent className="w-full max-w-xs sm:max-w-sm">
                            {availableDevices.map((dev) => (
                              <DropdownMenuItem
                                key={`${dev.device}-${dev.device_index}`}
                                onClick={() => handleDeviceChange(dev.device)}
                                className={selectedDevice === dev.device ? "bg-accent" : ""}
                              >
                                <div className="flex items-center justify-between w-full gap-2">
                                  <div className="flex items-center gap-2">
                                    <Badge variant={dev.device === "cuda" ? "default" : dev.device === "mps" ? "secondary" : "outline"}
                                      className="text-[10px] px-1.5 py-0">
                                      {dev.device.toUpperCase()}
                                    </Badge>
                                    <span className="text-sm">{dev.name}</span>
                                  </div>
                                  {dev.memory_gb > 0 && (
                                    <span className="text-xs text-muted-foreground shrink-0">{dev.memory_gb} GB</span>
                                  )}
                                </div>
                              </DropdownMenuItem>
                            ))}
                          </DropdownMenuContent>
                        </DropdownMenu>

                        {paramsAdjusted && (
                          <p className="text-[11px] text-muted-foreground flex items-center gap-1">
                            <CheckCircle2 className="h-3 w-3 text-green-500" />
                            Parameters auto-adjusted for {(selectedDevice || "cpu").toUpperCase()}
                            {(() => {
                              const dev = availableDevices.find(d => d.device === selectedDevice)
                              return dev && dev.memory_gb > 0 ? ` (${dev.memory_gb} GB)` : ""
                            })()}
                          </p>
                        )}
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Right: Hyperparameters */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg sm:text-xl">Training Parameters</CardTitle>
                  <CardDescription className="text-sm">Adjust hyperparameters (sensible defaults provided)</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
                    {([
                      ["epochs", "Epochs", "number"],
                      ["batchSize", "Batch Size", "number"],
                      ["learningRate", "Learning Rate", "text"],
                      ["maxSeqLength", "Max Seq Length", "number"],
                      ["loraR", "LoRA Rank (r)", "number"],
                      ["loraAlpha", "LoRA Alpha", "number"],
                      ["gradientAccumulationSteps", "Grad. Accum. Steps", "number"],
                      ["warmupRatio", "Warmup Ratio", "text"],
                      ["weightDecay", "Weight Decay", "text"],
                      ["saveSteps", "Save Steps", "number"],
                    ] as const).map(([key, label, type]) => (
                      <div key={key} className="space-y-1.5">
                        <Label className="text-xs">{label}</Label>
                        <Input
                          type={type}
                          value={params[key as keyof typeof params]}
                          onChange={(e) => {
                            const v = type === "number" ? Number(e.target.value) : e.target.value
                            setParam(key as keyof typeof DEFAULT_PARAMS, v as never)
                          }}
                          className="h-8 sm:h-9 text-xs sm:text-sm touch-manipulation"
                        />
                      </div>
                    ))}
                  </div>

                  <div className="mt-4 sm:mt-6 flex flex-col sm:flex-row gap-2 sm:gap-3">
                    <Button
                      onClick={startTraining}
                      disabled={isTraining || !csvFile || !params.baseModel.trim() || !serverRunning}
                      className="flex-1 h-9 sm:h-10 text-sm active:scale-95 touch-manipulation"
                    >
                      {isTraining ? (
                        <><Loader2 className="mr-2 h-3 w-3 sm:h-4 sm:w-4 animate-spin" /><span className="text-xs sm:text-sm">Training…</span></>
                      ) : (
                        <><Play className="mr-2 h-3 w-3 sm:h-4 sm:w-4" /><span className="text-xs sm:text-sm">Start Training</span></>
                      )}
                    </Button>
                    {isTraining && (
                      <Button variant="destructive" onClick={stopTraining} className="flex-1 sm:flex-initial h-9 sm:h-10 text-sm active:scale-95 touch-manipulation">
                        <Square className="mr-2 h-3 w-3 sm:h-4 sm:w-4" />
                        <span className="text-xs sm:text-sm">Stop</span>
                      </Button>
                    )}
                  </div>

                  {error && (
                    <div className="mt-3 flex items-start gap-2 text-sm text-destructive">
                      <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
                      <span>{error}</span>
                    </div>
                  )}

                  {!serverRunning && !isTraining && (
                    <div className="mt-3 flex items-start gap-2 text-sm text-amber-600 dark:text-amber-400">
                      <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
                      <span>Start the Python Training Server above before training.</span>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* ═══════════════ TAB 2: MONITOR ═══════════════ */}
          <TabsContent value="monitor">
            {/* ── Pipeline Stepper ── */}
            <Card className="mb-6">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg">Training Pipeline</CardTitle>
                <CardDescription>Step-by-step progress through the training process</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative">
                  {pipelineSteps.map((step, i) => {
                    const isLast = i === pipelineSteps.length - 1
                    return (
                      <div key={step.id} className="flex gap-3">
                        {/* Vertical timeline */}
                        <div className="flex flex-col items-center">
                          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border-2 transition-colors duration-300"
                            style={{
                              borderColor:
                                step.status === "completed" ? "hsl(var(--primary))" :
                                step.status === "in-progress" ? "hsl(var(--chart-2))" :
                                step.status === "error" ? "hsl(var(--destructive))" :
                                "hsl(var(--border))",
                              backgroundColor:
                                step.status === "completed" ? "hsl(var(--primary))" :
                                "transparent",
                            }}>
                            {step.status === "completed" ? (
                              <CheckCircle2 className="h-4 w-4 text-primary-foreground" />
                            ) : step.status === "in-progress" ? (
                              <Loader2 className="h-4 w-4 animate-spin" style={{ color: "hsl(var(--chart-2))" }} />
                            ) : step.status === "error" ? (
                              <AlertCircle className="h-4 w-4 text-destructive" />
                            ) : (
                              <Circle className="h-3 w-3 text-muted-foreground" />
                            )}
                          </div>
                          {!isLast && (
                            <div className="w-0.5 grow my-1 transition-colors duration-300"
                              style={{
                                backgroundColor:
                                  step.status === "completed" ? "hsl(var(--primary))" :
                                  "hsl(var(--border))",
                                minHeight: "1.25rem",
                              }} />
                          )}
                        </div>

                        {/* Step content */}
                        <div className={`pb-4 ${isLast ? "pb-0" : ""}`}>
                          <div className={`text-sm font-medium leading-7 ${
                            step.status === "completed" ? "text-foreground" :
                            step.status === "in-progress" ? "text-foreground" :
                            step.status === "error" ? "text-destructive" :
                            "text-muted-foreground"
                          }`}>
                            {step.label}
                            {step.status === "completed" && (
                              <Badge variant="secondary" className="ml-2 text-[10px] px-1.5 py-0 font-normal bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/20">Done</Badge>
                            )}
                            {step.status === "in-progress" && (
                              <Badge variant="secondary" className="ml-2 text-[10px] px-1.5 py-0 font-normal bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20 animate-pulse">In Progress</Badge>
                            )}
                          </div>
                          {step.detail && (
                            <p className="text-xs text-muted-foreground mt-0.5">{step.detail}</p>
                          )}

                          {/* Download progress bar inside the download_model step */}
                          {step.id === "download_model" && step.status === "in-progress" && downloadProgress && downloadProgress.percent < 100 && (
                            <div className="mt-2 space-y-1.5 max-w-md">
                              <Progress value={downloadProgress.percent} className="h-2.5 w-full" />
                              <div className="flex items-center justify-between text-[11px] text-muted-foreground font-mono">
                                <span className="flex items-center gap-1.5">
                                  <Download className="h-3 w-3" />
                                  {downloadProgress.filename ? downloadProgress.filename.slice(0, 30) : "Downloading…"}
                                </span>
                                <span>
                                  {downloadProgress.percent.toFixed(1)}%{" · "}
                                  {formatBytes(downloadProgress.speedBytesPerSec)}/s{" · "}
                                  {formatBytes(downloadProgress.downloadedBytes)} / {formatBytes(downloadProgress.totalBytes)}
                                </span>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              {/* Progress */}
              <Card>
                <CardHeader>
                  <CardTitle>Training Progress</CardTitle>
                  <CardDescription>{statusMessage || "Waiting to start…"}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Progress value={progressPercent} className="w-full" />
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <p className="text-2xl font-bold">{currentStep}</p>
                      <p className="text-xs text-muted-foreground">Step</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold">{totalSteps}</p>
                      <p className="text-xs text-muted-foreground">Total Steps</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold">
                        {metricsHistory.length > 0
                          ? metricsHistory[metricsHistory.length - 1].epoch
                          : 0}
                      </p>
                      <p className="text-xs text-muted-foreground">Epoch</p>
                    </div>
                  </div>

                  {metricsHistory.length > 0 && (
                    <div className="grid grid-cols-2 gap-4 rounded-md border border-border p-3">
                      <div>
                        <p className="text-xs text-muted-foreground">Latest Loss</p>
                        <p className="text-lg font-mono font-semibold">
                          {metricsHistory[metricsHistory.length - 1].loss?.toFixed(4) ?? "–"}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Learning Rate</p>
                        <p className="text-lg font-mono font-semibold">
                          {metricsHistory[metricsHistory.length - 1].learning_rate?.toExponential(2) ?? "–"}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Elapsed</p>
                        <p className="text-lg font-mono font-semibold">
                          {Math.round(metricsHistory[metricsHistory.length - 1].elapsed_seconds)}s
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">CPU / RAM</p>
                        <p className="text-lg font-mono font-semibold">
                          {metricsHistory[metricsHistory.length - 1].system.cpu_percent}% / {metricsHistory[metricsHistory.length - 1].system.ram_percent}%
                        </p>
                      </div>
                    </div>
                  )}

                  {trainingDone && outputDir && (
                    <div className="rounded-md border border-green-500/30 bg-green-500/10 p-3 space-y-3">
                      <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                        <CheckCircle2 className="h-4 w-4" />
                        <span className="text-sm font-medium">Training Complete!</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Input
                          placeholder="Model name for Ollama"
                          value={ollamaModelName}
                          onChange={(e) => setOllamaModelName(e.target.value)}
                          className="h-9 text-sm"
                        />
                        <Button size="sm" onClick={saveToOllama} disabled={saving || !ollamaModelName.trim()}>
                          {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4 mr-1" />}
                          Save
                        </Button>
                      </div>
                      {saveResult && (
                        <p className={`text-xs ${saveResult.startsWith("Error") ? "text-destructive" : "text-green-600 dark:text-green-400"}`}>
                          {saveResult}
                        </p>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Logs */}
              <Card>
                <CardHeader>
                  <CardTitle>Training Logs</CardTitle>
                  <CardDescription>{logs.length} log entries</CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[400px] rounded-md border border-border bg-muted/30 p-3">
                    <div className="space-y-1 font-mono text-xs">
                      {logs.length === 0 ? (
                        <p className="text-muted-foreground">Logs will appear here when training starts…</p>
                      ) : (
                        logs.map((log, i) => (
                          <div key={i} className="flex gap-2">
                            <span className="text-muted-foreground shrink-0">[{log.timestamp}]</span>
                            <span className="break-all">{log.message}</span>
                          </div>
                        ))
                      )}
                      <div ref={logsEndRef} />
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* ═══════════════ TAB 3: CHARTS ═══════════════ */}
          <TabsContent value="charts">
            <div className="grid gap-6 lg:grid-cols-2">
              {/* Loss Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>Training Loss</CardTitle>
                  <CardDescription>Loss over training steps</CardDescription>
                </CardHeader>
                <CardContent>
                  {lossData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={lossData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                        <XAxis dataKey="step" tick={{ fontSize: 11 }} className="text-muted-foreground" />
                        <YAxis tick={{ fontSize: 11 }} className="text-muted-foreground" />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: 8,
                            fontSize: 12,
                          }}
                        />
                        <Line type="monotone" dataKey="loss" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex items-center justify-center h-[300px] text-muted-foreground text-sm">
                      No data yet
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Learning Rate Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>Learning Rate Schedule</CardTitle>
                  <CardDescription>Learning rate over training steps</CardDescription>
                </CardHeader>
                <CardContent>
                  {lossData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={lossData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                        <XAxis dataKey="step" tick={{ fontSize: 11 }} />
                        <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => v.toExponential(1)} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: 8,
                            fontSize: 12,
                          }}
                        />
                        <Line type="monotone" dataKey="lr" stroke="hsl(var(--chart-2))" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex items-center justify-center h-[300px] text-muted-foreground text-sm">
                      No data yet
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* CPU / RAM Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>System Usage</CardTitle>
                  <CardDescription>CPU and RAM utilization</CardDescription>
                </CardHeader>
                <CardContent>
                  {systemData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={systemData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                        <XAxis dataKey="step" tick={{ fontSize: 11 }} />
                        <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} unit="%" />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: 8,
                            fontSize: 12,
                          }}
                        />
                        <Area type="monotone" dataKey="cpu" stroke="hsl(var(--chart-1))" fill="hsl(var(--chart-1))" fillOpacity={0.2} strokeWidth={2} name="CPU %" />
                        <Area type="monotone" dataKey="ram" stroke="hsl(var(--chart-3))" fill="hsl(var(--chart-3))" fillOpacity={0.2} strokeWidth={2} name="RAM %" />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex items-center justify-center h-[300px] text-muted-foreground text-sm">
                      No data yet
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* GPU Memory Chart (only when gpu data exists) */}
              <Card>
                <CardHeader>
                  <CardTitle>GPU Memory</CardTitle>
                  <CardDescription>GPU memory utilization during training</CardDescription>
                </CardHeader>
                <CardContent>
                  {systemData.some(d => d.gpuMem > 0) ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={systemData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                        <XAxis dataKey="step" tick={{ fontSize: 11 }} />
                        <YAxis tick={{ fontSize: 11 }} unit=" GB" />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: 8,
                            fontSize: 12,
                          }}
                        />
                        <Area type="monotone" dataKey="gpuMem" stroke="hsl(var(--chart-4))" fill="hsl(var(--chart-4))" fillOpacity={0.2} strokeWidth={2} name="GPU Mem (GB)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-[300px] text-muted-foreground text-sm gap-2">
                      <Cpu className="h-8 w-8 opacity-50" />
                      {deviceInfo?.device === "cuda" ? "Waiting for GPU data…" : "No GPU detected — training on CPU/MPS"}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* ═══════════════ TAB 4: TEST ═══════════════ */}
          <TabsContent value="test">
            <Card>
              <CardHeader>
                <CardTitle>Test Your Model</CardTitle>
                <CardDescription>
                  Chat with your trained model using the full RAG pipeline
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex flex-wrap gap-3">
                  {/* Model selector */}
                  <div className="space-y-1.5 min-w-[200px]">
                    <Label className="text-xs">Model</Label>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" className="w-full justify-between text-sm h-9">
                          <div className="flex items-center gap-2">
                            <Cpu className="h-3.5 w-3.5" />
                            <span className="truncate">{testModel || "Select model"}</span>
                          </div>
                          <ChevronDown className="h-3.5 w-3.5 ml-2" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="max-h-64 overflow-y-auto">
                        {availableModels.map((m) => (
                          <DropdownMenuItem key={m.name} onClick={() => setTestModel(m.name)}
                            className={testModel === m.name ? "bg-accent" : ""}>
                            <Cpu className="mr-2 h-4 w-4" />{m.name}
                          </DropdownMenuItem>
                        ))}
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>

                  {/* Document selector for RAG */}
                  {documents.length > 0 && (
                    <div className="space-y-1.5 min-w-[200px]">
                      <Label className="text-xs">Document (RAG)</Label>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="outline" className="w-full justify-between text-sm h-9">
                            <div className="flex items-center gap-2">
                              <FileText className="h-3.5 w-3.5" />
                              <span className="truncate">{testDocument || "None"}</span>
                            </div>
                            <ChevronDown className="h-3.5 w-3.5 ml-2" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent className="max-h-64 overflow-y-auto">
                          <DropdownMenuItem onClick={() => setTestDocument(null)} className={!testDocument ? "bg-accent" : ""}>
                            None
                          </DropdownMenuItem>
                          {documents.map((d) => (
                            <DropdownMenuItem key={d} onClick={() => setTestDocument(d)}
                              className={testDocument === d ? "bg-accent" : ""}>
                              <FileText className="mr-2 h-4 w-4" />{d}
                            </DropdownMenuItem>
                          ))}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  )}
                </div>

                {/* Chat area */}
                <ScrollArea className="h-[350px] rounded-md border border-border bg-muted/20 p-4">
                  <div className="space-y-4">
                    {testMessages.length === 0 ? (
                      <div className="flex flex-col items-center justify-center py-16 text-center text-muted-foreground">
                        <MessageSquare className="h-10 w-10 opacity-50 mb-2" />
                        <p className="text-sm">Send a message to test your trained model</p>
                      </div>
                    ) : (
                      testMessages.map((msg) => (
                        <div key={msg.id} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                          {msg.role === "assistant" && (
                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                              <Bot className="h-4 w-4" />
                            </div>
                          )}
                          <div className={`max-w-[75%] rounded-xl px-3 py-2 text-sm ${
                            msg.role === "user"
                              ? "bg-primary text-primary-foreground"
                              : "bg-card border border-border"
                          }`}>
                            <p className="whitespace-pre-wrap">{msg.content}</p>
                          </div>
                          {msg.role === "user" && (
                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-secondary text-secondary-foreground">
                              <User className="h-4 w-4" />
                            </div>
                          )}
                        </div>
                      ))
                    )}
                    {testLoading && (
                      <div className="flex gap-3">
                        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                          <Bot className="h-4 w-4" />
                        </div>
                        <div className="rounded-xl border border-border bg-card px-3 py-2">
                          <Loader2 className="h-4 w-4 animate-spin" />
                        </div>
                      </div>
                    )}
                  </div>
                </ScrollArea>

                <div className="flex gap-2">
                  <Input
                    placeholder="Ask a question…"
                    value={testInput}
                    onChange={(e) => setTestInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendTestMessage() } }}
                    disabled={!testModel || testLoading}
                    className="flex-1"
                  />
                  <Button onClick={sendTestMessage} disabled={!testModel || !testInput.trim() || testLoading} size="icon">
                    {testLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

export default function TrainingPage() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
      <TrainingContent />
    </ThemeProvider>
  )
}
