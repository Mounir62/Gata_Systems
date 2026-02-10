import { NextRequest, NextResponse } from "next/server"
import { buildSystemPrompt } from "@/lib/prompts"

export async function POST(request: NextRequest) {
  try {
    const { model, messages, selectedDocument, complexity = "balanced" } = await request.json()

    const ollamaUrl = process.env.OLLAMA_URL || "http://localhost:11434"

    // Define max_tokens and instructions based on complexity level
    const complexityConfig = {
      simple: {
        max_tokens: 500,
        instruction: "Provide a brief, concise answer. Keep it short and to the point."
      },
      balanced: {
        max_tokens: 700,
        instruction: "Provide a clear and informative answer with appropriate detail."
      },
      detailed: {
        max_tokens: 1400,
        instruction: "Provide a comprehensive and detailed answer with thorough explanations, examples, and relevant context."
      }
    }

    const config = complexityConfig[complexity as keyof typeof complexityConfig] || complexityConfig.balanced

    // Build the system prompt with optional RAG context
    let systemPrompt = ""
    if (selectedDocument && messages.length > 0) {
      const lastUserMessage = messages[messages.length - 1]
      if (lastUserMessage.role === "user") {
        systemPrompt = await buildSystemPrompt(
          lastUserMessage.content,
          ollamaUrl,
          selectedDocument
        )
      }
    } else {
      systemPrompt = await buildSystemPrompt("", ollamaUrl)
    }

    // Append complexity instruction to system prompt
    systemPrompt += "\n\n" + config.instruction

    // Add the system prompt to the beginning of the messages array
    const ollamaMessages = [
      { role: "system", content: systemPrompt },
      ...messages.map((msg: { role: string; content: string }) => ({
        role: msg.role,
        content: msg.content,
      })),
    ]

    const response = await fetch(`${ollamaUrl}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: ollamaMessages,
        stream: true,
        temperature: 0.7,
        max_tokens: config.max_tokens,
      }),
    })

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`)
    }

    // Create a TransformStream to process the streaming response
    const encoder = new TextEncoder()
    const decoder = new TextDecoder()

    const transformStream = new TransformStream({
      async transform(chunk, controller) {
        const text = decoder.decode(chunk)
        const lines = text.split("\n").filter((line) => line.trim())

        for (const line of lines) {
          try {
            const json = JSON.parse(line)
            if (json.message?.content) {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content: json.message.content })}\n\n`))
            }
            if (json.done) {
              controller.enqueue(encoder.encode(`data: [DONE]\n\n`))
            }
          } catch {
            // Skip invalid JSON lines
          }
        }
      },
    })

    return new Response(response.body?.pipeThrough(transformStream), {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    })
  } catch (error) {
    console.error("Ollama API error:", error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to connect to Ollama" },
      { status: 500 }
    )
  }
}
