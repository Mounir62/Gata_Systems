import { NextResponse } from "next/server"

export async function GET() {
  try {
    const ollamaUrl = process.env.OLLAMA_URL || "http://localhost:11434"

    const response = await fetch(`${ollamaUrl}/api/tags`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`)
    }

    const data = await response.json()

    // Extract model names from the response
    const models = data.models?.map((model: { name: string; modified_at: string; size: number }) => ({
      name: model.name,
      modified_at: model.modified_at,
      size: model.size,
    })) || []

    return NextResponse.json({ models })
  } catch (error) {
    console.error("Failed to fetch Ollama models:", error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to fetch models from Ollama" },
      { status: 500 }
    )
  }
}
