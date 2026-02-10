import { NextResponse } from "next/server"
import { listDocuments, deleteDocument } from "@/lib/vector-store"

export async function GET() {
  try {
    const documents = listDocuments()
    return NextResponse.json({ documents })
  } catch (error) {
    console.error("List documents error:", error)
    return NextResponse.json({ documents: [] })
  }
}

export async function DELETE(request: Request) {
  try {
    const { filename } = await request.json()
    if (!filename) {
      return NextResponse.json({ error: "filename is required" }, { status: 400 })
    }
    deleteDocument(filename)
    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("Delete document error:", error)
    return NextResponse.json(
      { error: "Failed to delete document" },
      { status: 500 }
    )
  }
}
