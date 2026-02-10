export interface TextChunk {
  id: string
  content: string
  metadata: {
    filename: string
    chunkIndex: number
    totalChunks: number
  }
}

/**
 * Split text into overlapping chunks for embedding.
 */
export function chunkText(
  text: string,
  filename: string,
  chunkSize = 800,
  chunkOverlap = 200
): TextChunk[] {
  // Normalise whitespace
  const cleaned = text.replace(/\s+/g, " ").trim()

  if (cleaned.length === 0) return []

  const chunks: TextChunk[] = []
  let start = 0

  while (start < cleaned.length) {
    const end = Math.min(start + chunkSize, cleaned.length)
    const content = cleaned.slice(start, end).trim()

    if (content.length > 0) {
      chunks.push({
        id: `${filename}-chunk-${chunks.length}`,
        content,
        metadata: {
          filename,
          chunkIndex: chunks.length,
          totalChunks: 0, // filled in below
        },
      })
    }

    // Move forward by (chunkSize - overlap)
    start += chunkSize - chunkOverlap
    if (start >= cleaned.length) break
  }

  // Back-fill totalChunks
  for (const c of chunks) {
    c.metadata.totalChunks = chunks.length
  }

  return chunks
}
