import pdf from "pdf-parse"

export interface PdfResult {
  text: string
  numPages: number
  filename: string
}

/**
 * Extract text content from a PDF buffer.
 */
export async function extractTextFromPdf(
  buffer: Buffer,
  filename: string
): Promise<PdfResult> {
  const data = await pdf(buffer)
  return {
    text: data.text,
    numPages: data.numpages,
    filename,
  }
}
