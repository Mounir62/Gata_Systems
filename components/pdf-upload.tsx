"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { FileUp, X, FileText, Loader2, Check } from "lucide-react"

interface PdfUploadProps {
  selectedDocument: string | null
  onDocumentChange: (filename: string | null) => void
}

export function PdfUpload({ selectedDocument, onDocumentChange }: PdfUploadProps) {
  const [isUploading, setIsUploading] = React.useState(false)
  const [uploadError, setUploadError] = React.useState<string | null>(null)
  const [uploadSuccess, setUploadSuccess] = React.useState<string | null>(null)
  const [documents, setDocuments] = React.useState<string[]>([])
  const fileInputRef = React.useRef<HTMLInputElement>(null)

  // Fetch existing documents on mount
  React.useEffect(() => {
    fetchDocuments()
  }, [])

  const fetchDocuments = async () => {
    try {
      const res = await fetch("/api/documents/list")
      if (res.ok) {
        const data = await res.json()
        setDocuments(data.documents || [])
      }
    } catch {
      // ignore
    }
  }

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (file.type !== "application/pdf") {
      setUploadError("Please select a PDF file.")
      return
    }

    setIsUploading(true)
    setUploadError(null)
    setUploadSuccess(null)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const res = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      })

      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.error || "Upload failed")
      }

      setUploadSuccess(`"${data.filename}" uploaded (${data.chunks} chunks from ${data.pages} pages)`)
      onDocumentChange(data.filename)
      await fetchDocuments()
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed")
    } finally {
      setIsUploading(false)
      // Reset file input
      if (fileInputRef.current) fileInputRef.current.value = ""
    }
  }

  const handleRemoveDocument = async (filename: string) => {
    try {
      await fetch("/api/documents/list", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename }),
      })
      if (selectedDocument === filename) {
        onDocumentChange(null)
      }
      await fetchDocuments()
    } catch {
      // ignore
    }
  }

  const clearSelection = () => {
    onDocumentChange(null)
    setUploadSuccess(null)
  }

  return (
    <div className="flex flex-col gap-2">
      {/* Upload button */}
      <div className="flex items-center gap-2">
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf"
          onChange={handleUpload}
          className="hidden"
          id="pdf-upload"
        />
        <Button
          variant="outline"
          size="sm"
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          className="flex items-center gap-2 text-xs"
        >
          {isUploading ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              <span>Processing…</span>
            </>
          ) : (
            <>
              <FileUp className="h-3.5 w-3.5" />
              <span>Upload PDF</span>
            </>
          )}
        </Button>

        {/* Currently active document badge */}
        {selectedDocument && (
          <div className="flex items-center gap-1 rounded-md border border-primary/30 bg-primary/10 px-2 py-1 text-xs text-primary">
            <FileText className="h-3 w-3" />
            <span className="max-w-[140px] truncate">{selectedDocument}</span>
            <button onClick={clearSelection} className="ml-1 hover:text-destructive">
              <X className="h-3 w-3" />
            </button>
          </div>
        )}
      </div>

      {/* Success message */}
      {uploadSuccess && !selectedDocument && (
        <p className="text-xs text-green-500 flex items-center gap-1">
          <Check className="h-3 w-3" /> {uploadSuccess}
        </p>
      )}

      {/* Error message */}
      {uploadError && (
        <p className="text-xs text-destructive">{uploadError}</p>
      )}

      {/* Previously uploaded documents */}
      {documents.length > 0 && !selectedDocument && (
        <div className="flex flex-wrap gap-1 mt-1">
          {documents.map((doc) => (
            <button
              key={doc}
              onClick={() => onDocumentChange(doc)}
              className="flex items-center gap-1 rounded-md border border-border bg-card px-2 py-1 text-xs text-muted-foreground hover:border-primary hover:text-primary transition-colors"
            >
              <FileText className="h-3 w-3" />
              <span className="max-w-[120px] truncate">{doc}</span>
              <span
                role="button"
                onClick={(e) => {
                  e.stopPropagation()
                  handleRemoveDocument(doc)
                }}
                className="ml-1 hover:text-destructive"
              >
                <X className="h-3 w-3" />
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
