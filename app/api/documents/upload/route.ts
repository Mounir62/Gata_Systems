import { NextRequest, NextResponse } from "next/server"
import { extractTextFromPdf } from "@/lib/pdf-processor"
import { chunkText } from "@/lib/chunker"
import { addChunks } from "@/lib/vector-store"

export const runtime = "nodejs"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File | null

    if (!file || file.type !== "application/pdf") {
      return NextResponse.json(
        { error: "Please upload a valid PDF file." },
        { status: 400 }
      )
    }

    const ollamaUrl = process.env.OLLAMA_URL || "http://localhost:11434"

    // 1. Read file into a buffer
    const arrayBuffer = await file.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)

    // 2. Extract text from PDF
    const pdfResult = await extractTextFromPdf(buffer, file.name)

    if (!pdfResult.text.trim()) {
      return NextResponse.json(
        { error: "Could not extract text from the PDF. It may be image-based." },
        { status: 422 }
      )
    }

    // 3. Chunk the text
    const chunks = chunkText(pdfResult.text, file.name)

    // 4. Embed & store chunks
    await addChunks(
      chunks.map((c) => ({
        id: c.id,
        content: c.content,
        metadata: c.metadata as unknown as Record<string, unknown>,
      })),
      ollamaUrl
    )

    return NextResponse.json({
      success: true,
      filename: file.name,
      pages: pdfResult.numPages,
      chunks: chunks.length,
    })
  } catch (error) {
    console.error("Upload error:", error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Upload failed" },
      { status: 500 }
    )
  }
}
