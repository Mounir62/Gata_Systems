import { NextResponse } from "next/server"
import { spawn, type ChildProcess } from "child_process"
import path from "path"

const PYTHON_SERVER_PORT = 8000

// Global handle on the server process
let serverProcess: ChildProcess | null = null
let serverLogs: string[] = []

/**
 * GET  → returns the server status (running / stopped) + recent logs
 */
export async function GET() {
  // Fast health-check: try reaching the server
  const running = await isServerHealthy()
  return NextResponse.json({
    running,
    pid: serverProcess?.pid ?? null,
    port: PYTHON_SERVER_PORT,
    logs: serverLogs.slice(-30), // last 30 log lines
  })
}

/**
 * POST → start the Python training server
 */
export async function POST() {
  // If already running, just confirm
  if (serverProcess && !serverProcess.killed) {
    const healthy = await isServerHealthy()
    if (healthy) {
      return NextResponse.json({
        running: true,
        pid: serverProcess.pid,
        port: PYTHON_SERVER_PORT,
        message: "Server is already running.",
      })
    }
    // Process exists but unhealthy – kill it and restart
    serverProcess.kill("SIGTERM")
    serverProcess = null
  }

  try {
    const projectRoot = process.cwd()

    // Try venv python first, then fall back to python3
    const venvPython = path.join(projectRoot, ".venv", "bin", "python")
    const fs = await import("fs")
    const pythonBin = fs.existsSync(venvPython) ? venvPython : "python3"

    const scriptPath = path.join(projectRoot, "scripts", "training_server.py")

    serverLogs = []

    const proc = spawn(pythonBin, [scriptPath, "--port", String(PYTHON_SERVER_PORT)], {
      cwd: projectRoot,
      env: { ...process.env },
      stdio: ["ignore", "pipe", "pipe"],
    })

    serverProcess = proc

    const appendLog = (text: string) => {
      const lines = text.split("\n").filter((l) => l.trim())
      serverLogs.push(...lines)
      // Keep only last 200 lines
      if (serverLogs.length > 200) {
        serverLogs = serverLogs.slice(-200)
      }
    }

    proc.stdout?.on("data", (chunk: Buffer) => {
      appendLog(chunk.toString())
    })

    proc.stderr?.on("data", (chunk: Buffer) => {
      appendLog(chunk.toString())
    })

    proc.on("close", (code) => {
      appendLog(`[server] Process exited with code ${code}`)
      serverProcess = null
    })

    proc.on("error", (err) => {
      appendLog(`[server] Error: ${err.message}`)
      serverProcess = null
    })

    // Wait a moment for the server to boot, then health check
    const healthy = await waitForServer(8, 500) // up to 4 seconds
    if (healthy) {
      return NextResponse.json({
        running: true,
        pid: proc.pid,
        port: PYTHON_SERVER_PORT,
        message: "Training server started successfully.",
      })
    }

    return NextResponse.json({
      running: false,
      pid: proc.pid,
      port: PYTHON_SERVER_PORT,
      message: "Server process started but health check is not responding yet. It may still be loading.",
      logs: serverLogs.slice(-10),
    })
  } catch (error) {
    console.error("Failed to start training server:", error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Failed to start server",
        logs: serverLogs.slice(-10),
      },
      { status: 500 }
    )
  }
}

/**
 * DELETE → stop the Python training server
 */
export async function DELETE() {
  if (serverProcess && !serverProcess.killed) {
    serverProcess.kill("SIGTERM")
    // Give it a moment to die
    await new Promise((r) => setTimeout(r, 500))
    if (serverProcess && !serverProcess.killed) {
      serverProcess.kill("SIGKILL")
    }
    serverProcess = null
    return NextResponse.json({ running: false, message: "Server stopped." })
  }
  return NextResponse.json({ running: false, message: "Server was not running." })
}

// ── Helpers -------------------------------------------------------------- //

async function isServerHealthy(): Promise<boolean> {
  try {
    const res = await fetch(`http://localhost:${PYTHON_SERVER_PORT}/health`, {
      signal: AbortSignal.timeout(2000),
    })
    return res.ok
  } catch {
    return false
  }
}

async function waitForServer(retries: number, delayMs: number): Promise<boolean> {
  for (let i = 0; i < retries; i++) {
    await new Promise((r) => setTimeout(r, delayMs))
    if (await isServerHealthy()) return true
  }
  return false
}
