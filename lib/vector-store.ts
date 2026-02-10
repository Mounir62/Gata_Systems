import fs from "fs"
import path from "path"

export interface StoredChunk {
  id: string
  content: string
  embedding: number[]
  metadata: Record<string, unknown>
}

export interface SearchResult {
  id: string
  content: string
  score: number
  metadata: Record<string, unknown>
}

const STORE_DIR = path.join(process.cwd(), "vector_store")
const STORE_FILE = path.join(STORE_DIR, "documents.json")

/** Cosine similarity between two vectors. */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0
  let magA = 0
  let magB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    magA += a[i] * a[i]
    magB += b[i] * b[i]
  }
  magA = Math.sqrt(magA)
  magB = Math.sqrt(magB)
  if (magA === 0 || magB === 0) return 0
  return dot / (magA * magB)
}

/** Load all stored chunks from disk. */
function loadStore(): StoredChunk[] {
  try {
    if (fs.existsSync(STORE_FILE)) {
      const raw = fs.readFileSync(STORE_FILE, "utf-8")
      return JSON.parse(raw) as StoredChunk[]
    }
  } catch {
    // corrupted file, start fresh
  }
  return []
}

/** Persist chunks to disk. */
function saveStore(chunks: StoredChunk[]): void {
  if (!fs.existsSync(STORE_DIR)) {
    fs.mkdirSync(STORE_DIR, { recursive: true })
  }
  fs.writeFileSync(STORE_FILE, JSON.stringify(chunks), "utf-8")
}

/** Get an embedding from Ollama. */
async function getEmbedding(
  text: string,
  ollamaUrl: string
): Promise<number[]> {
  const res = await fetch(`${ollamaUrl}/api/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "nomic-embed-text", input: text }),
  })
  if (!res.ok) {
    throw new Error(`Ollama embedding error: ${res.statusText}`)
  }
  const data = await res.json()
  // Ollama returns { embeddings: [[...]] } for /api/embed
  return data.embeddings[0]
}

/**
 * Add chunks to the vector store.
 * Generates embeddings via Ollama and persists to disk.
 */
export async function addChunks(
  chunks: { id: string; content: string; metadata: Record<string, unknown> }[],
  ollamaUrl: string
): Promise<void> {
  const store = loadStore()

  for (const chunk of chunks) {
    const embedding = await getEmbedding(chunk.content, ollamaUrl)
    store.push({
      id: chunk.id,
      content: chunk.content,
      embedding,
      metadata: chunk.metadata,
    })
  }

  saveStore(store)
}

/**
 * Search the vector store for chunks similar to the query.
 * Returns top-k results filtered to a specific filename if provided.
 */
export async function searchChunks(
  query: string,
  ollamaUrl: string,
  options: { topK?: number; filename?: string } = {}
): Promise<SearchResult[]> {
  const { topK = 5, filename } = options
  const store = loadStore()
  if (store.length === 0) return []

  const queryEmbedding = await getEmbedding(query, ollamaUrl)

  let candidates = store
  if (filename) {
    candidates = store.filter((c) => c.metadata.filename === filename)
  }

  const scored = candidates.map((chunk) => ({
    id: chunk.id,
    content: chunk.content,
    score: cosineSimilarity(queryEmbedding, chunk.embedding),
    metadata: chunk.metadata,
  }))

  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, topK)
}

/** List all unique document filenames in the store. */
export function listDocuments(): string[] {
  const store = loadStore()
  const names = new Set<string>()
  for (const chunk of store) {
    if (chunk.metadata.filename) {
      names.add(chunk.metadata.filename as string)
    }
  }
  return Array.from(names)
}

/** Delete all chunks for a given filename. */
export function deleteDocument(filename: string): void {
  const store = loadStore()
  const filtered = store.filter((c) => c.metadata.filename !== filename)
  saveStore(filtered)
}
