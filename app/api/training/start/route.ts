import { NextRequest, NextResponse } from "next/server"

const PYTHON_SERVER = "http://localhost:8000"

/**
 * POST – Proxy the training request to the Python FastAPI server.
 *        Forwards the multipart form (file + params) and streams back
 *        the SSE response to the browser.
 */
export async function POST(request: NextRequest) {
  try {
    // Quick health-check first
    try {
      const h = await fetch(`${PYTHON_SERVER}/health`, { signal: AbortSignal.timeout(2000) })
      if (!h.ok) throw new Error()
    } catch {
      return NextResponse.json(
        { error: "Training server is not running. Please start it first." },
        { status: 503 }
      )
    }

    const formData = await request.formData()

    // Forward the multipart form to the Python server
    const pyRes = await fetch(`${PYTHON_SERVER}/train`, {
      method: "POST",
      body: formData,
    })

    if (!pyRes.ok) {
      const data = await pyRes.json().catch(() => ({ error: "Unknown error" }))
      return NextResponse.json(
        { error: data.error || `Server responded with ${pyRes.status}` },
        { status: pyRes.status }
      )
    }

    // Stream the SSE response through to the client
    const body = pyRes.body
    if (!body) {
      return NextResponse.json({ error: "No stream from server" }, { status: 502 })
    }

    return new Response(body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    })
  } catch (error) {
    console.error("Training start error:", error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to start training" },
      { status: 500 }
    )
  }
}

/**
 * DELETE – Tell the Python server to cancel the running training job.
 */
export async function DELETE() {
  try {
    const res = await fetch(`${PYTHON_SERVER}/train/stop`, {
      method: "POST",
      signal: AbortSignal.timeout(5000),
    })
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch {
    return NextResponse.json({ error: "Could not reach training server." }, { status: 503 })
  }
}
