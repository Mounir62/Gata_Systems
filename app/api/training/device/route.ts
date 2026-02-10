import { NextResponse } from "next/server"

const PYTHON_SERVER = "http://localhost:8000"

export async function GET() {
  // Try the Python training server first (fast, no cold-start)
  try {
    const res = await fetch(`${PYTHON_SERVER}/device`, {
      signal: AbortSignal.timeout(3000),
    })
    if (res.ok) {
      const info = await res.json()
      return NextResponse.json(info)
    }
  } catch {
    // Server not running — fall back to direct Python call
  }

  // Fallback: quick Python one-liner
  try {
    const { execSync } = await import("child_process")
    const path = await import("path")
    const fs = await import("fs")
    const projectRoot = process.cwd()
    const venvPython = path.join(projectRoot, ".venv", "bin", "python")
    const pythonBin = fs.existsSync(venvPython) ? venvPython : "python3"

    const script = `
import json, sys
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
        print(json.dumps({"device": "cuda", "name": name, "memory_gb": mem}))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        import platform
        print(json.dumps({"device": "mps", "name": "Apple Silicon GPU (" + platform.processor() + ")", "memory_gb": 0}))
    else:
        print(json.dumps({"device": "cpu", "name": "CPU", "memory_gb": 0}))
except ImportError:
    print(json.dumps({"device": "cpu", "name": "CPU (PyTorch not installed)", "memory_gb": 0, "error": "PyTorch is not installed."}))
`
    const result = execSync(`${pythonBin} -c '${script.replace(/'/g, "'\\''")}'`, {
      timeout: 30000,
      encoding: "utf-8",
    })
    const info = JSON.parse(result.trim())
    return NextResponse.json(info)
  } catch (error) {
    console.error("Device detection error:", error)
    return NextResponse.json({
      device: "cpu",
      name: "CPU (detection failed)",
      memory_gb: 0,
      error: error instanceof Error ? error.message : "Unknown error",
    })
  }
}
