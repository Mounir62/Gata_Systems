import { NextRequest, NextResponse } from "next/server"
import { buildSystemPrompt, DEFAULT_SYSTEM_PROMPT } from "@/lib/prompts"

export async function POST(request: NextRequest) {
  try {
    const { questions, model, selectedDocument } = await request.json()

    if (!Array.isArray(questions) || questions.length === 0) {
      return NextResponse.json(
        { error: "Please provide an array of questions." },
        { status: 400 }
      )
    }

    if (!model) {
      return NextResponse.json(
        { error: "Please select a model." },
        { status: 400 }
      )
    }

    const ollamaUrl = process.env.OLLAMA_URL || "http://localhost:11434"

    // Create a streaming response
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      async start(controller) {
        const total = questions.filter((q) => q.trim()).length

        let processed = 0

        // Process each question sequentially
        for (const question of questions) {
          if (!question.trim()) continue

          try {
            // Build system prompt with RAG context if document is selected
            const systemPrompt = await buildSystemPrompt(
              question.trim(),
              ollamaUrl,
              selectedDocument
            )

            // Include full system prompt + user question in the question field
            const fullPrompt = `System Prompt:\n${systemPrompt}\n\nUser Question:\n${question.trim()}`

            const response = await fetch(`${ollamaUrl}/api/chat`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                model,
                messages: [
                  { role: "system", content: systemPrompt },
                  { role: "user", content: question.trim() },
                ],
                stream: false,
              }),
            })

            if (!response.ok) {
              throw new Error(`Ollama API error: ${response.statusText}`)
            }

            const data = await response.json()
            const answer = data.message?.content || "No response"

            processed++

            // Send progress update
            const progressData = {
              type: "progress",
              current: processed,
              total,
              result: {
                question: fullPrompt,
                answer: answer.trim(),
              },
            }

            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify(progressData)}\n\n`)
            )
          } catch (error) {
            console.error(`Error processing question: ${question}`, error)
            processed++

            const errorData = {
              type: "progress",
              current: processed,
              total,
              result: {
                question: question.trim(),
                answer: `Error: ${error instanceof Error ? error.message : "Failed to get response"}`,
              },
            }

            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify(errorData)}\n\n`)
            )
          }
        }

        // Send completion signal
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "done" })}\n\n`))
        controller.close()
      },
    })

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    })
  } catch (error) {
    console.error("Dataset generation error:", error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to generate dataset" },
      { status: 500 }
    )
  }
}
