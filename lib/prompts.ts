import { searchChunks } from "@/lib/vector-store"

export const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely and clearly, make sure to put emojis before headings."

/**
 * Build a system prompt, optionally with RAG context from a document.
 */
export async function buildSystemPrompt(
  userQuery: string,
  ollamaUrl: string,
  selectedDocument?: string | null
): Promise<string> {
  if (!selectedDocument) {
    return DEFAULT_SYSTEM_PROMPT
  }

  try {
    const results = await searchChunks(userQuery, ollamaUrl, {
      topK: 5,
      filename: selectedDocument,
    })

    if (results.length === 0) {
      return DEFAULT_SYSTEM_PROMPT
    }

    const context = results
      .map((r, i) => `[Context ${i + 1}]:\n${r.content}`)
      .join("\n\n")

    return `You are a knowledgeable and thorough assistant answering questions about the document "${selectedDocument}".

Use the following context from the document to answer the user's question. Your goal is to produce a complete, well-structured, and detailed answer that fully explains the relevant ideas found in the context. Elaborate on definitions, reasoning steps, implications, and relationships between concepts mentioned in the document.

If the context does not contain enough information to fully answer the question, clearly state that and explain what information is missing.

Context:
${context}

Instructions:
- Base your answer primarily on the provided context
- Provide an in-depth, explanatory response rather than a short summary
- Break down complex ideas into clear, logical parts
- Use examples or clarifications when supported by the context
- Maintain accuracy and avoid introducing unsupported information
- Use emojis frequently to improve readability, structure, and emphasis (e.g., headings, key points, warnings, summaries)
- Use emojis naturally and consistently throughout the answer
`
  } catch (error) {
    console.error("Error retrieving document context:", error)
    return DEFAULT_SYSTEM_PROMPT
  }
}
