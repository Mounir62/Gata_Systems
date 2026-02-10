import { NextResponse } from "next/server"

const PYTHON_SERVER = "http://localhost:8000"

/**
 * GET /api/training/devices
 * Returns all available training devices (CPU, MPS, CUDA, etc.)
 */
export async function GET() {
  // Try the Python training server first
  try {
    const res = await fetch(`${PYTHON_SERVER}/devices`, {
      signal: AbortSignal.timeout(5000),
    })
    if (res.ok) {
      const data = await res.json()
      return NextResponse.json(data)
    }
  } catch {
    // Server not running — fall back to direct Python call
  }

  // Fallback: detect via one-liner
  try {
    const { execSync } = await import("child_process")
    const path = await import("path")
    const fs = await import("fs")
    const projectRoot = process.cwd()
    const venvPython = path.join(projectRoot, ".venv", "bin", "python")
    const pythonBin = fs.existsSync(venvPython) ? venvPython : "python3"

    const script = `
import json, sys, os
devices = []
try:
    import psutil
    ram = round(psutil.virtual_memory().total / (1024**3), 1)
except Exception:
    ram = 0
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = round(torch.cuda.get_device_properties(i).total_mem / (1024**3), 1)
            devices.append({"device": "cuda", "device_index": i, "name": name, "memory_gb": mem})
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        import platform
        devices.append({"device": "mps", "device_index": 0, "name": "Apple Silicon GPU (" + platform.processor() + ")", "memory_gb": ram})
except ImportError:
    pass
devices.append({"device": "cpu", "device_index": 0, "name": "CPU", "memory_gb": ram})
print(json.dumps({"devices": devices}))
`
    const result = execSync(`${pythonBin} -c '${script.replace(/'/g, "'\\''")}'`, {
      timeout: 30000,
      encoding: "utf-8",
    })
    const data = JSON.parse(result.trim())
    return NextResponse.json(data)
  } catch (error) {
    console.error("Devices detection error:", error)
    return NextResponse.json({
      devices: [{ device: "cpu", device_index: 0, name: "CPU (detection failed)", memory_gb: 0 }],
    })
  }
}
