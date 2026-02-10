import { NextRequest, NextResponse } from "next/server"
import { execSync } from "child_process"
import path from "path"
import fs from "fs"

export async function POST(request: NextRequest) {
  try {
    const { outputDir, modelName } = await request.json()

    if (!outputDir || !modelName) {
      return NextResponse.json(
        { error: "outputDir and modelName are required." },
        { status: 400 }
      )
    }

    if (!fs.existsSync(outputDir)) {
      return NextResponse.json(
        { error: "Output directory does not exist." },
        { status: 404 }
      )
    }

    // Verify model files exist in the output directory
    const files = fs.readdirSync(outputDir)
    const hasGGUF = files.some(f => f.endsWith(".gguf"))
    const hasSafetensors = files.some(f => f.endsWith(".safetensors"))
    const hasConfig = files.some(f => f === "config.json")

    // Prefer GGUF (universal Ollama support), fallback to safetensors
    const modelPath = hasGGUF 
      ? path.join(outputDir, files.find(f => f.endsWith(".gguf"))!)
      : outputDir

    if (!hasGGUF && !hasSafetensors) {
      // Check if only adapter files exist (merge didn't happen)
      const hasAdapter = files.some(f => f === "adapter_config.json" || f === "adapter_model.safetensors")
      if (hasAdapter) {
        return NextResponse.json(
          { error: "Only LoRA adapter files found — the model was not merged. Please re-train." },
          { status: 400 }
        )
      }
      return NextResponse.json(
        { error: `No model files found in output directory. Files present: ${files.join(", ") || "(empty)"}` },
        { status: 400 }
      )
    }

    if (!hasGGUF && !hasConfig) {
      return NextResponse.json(
        { error: "config.json not found in output directory. The model may not have been saved correctly." },
        { status: 400 }
      )
    }

    // Build the Modelfile
    // If GGUF exists, use it directly (universal format)
    // Otherwise use safetensors directory (requires Ollama >= 0.5 with architecture support)
    const modelfileLines = [
      `FROM ${modelPath}`,
    ]

    // Read tokenizer_config.json for chat template (only if using safetensors directory)
    if (!hasGGUF) {
      const tokenizerConfigPath = path.join(outputDir, "tokenizer_config.json")
      if (fs.existsSync(tokenizerConfigPath)) {
        try {
          const tokConfig = JSON.parse(fs.readFileSync(tokenizerConfigPath, "utf-8"))
          if (tokConfig.chat_template) {
            // Ollama reads chat_template from tokenizer_config.json automatically
            // No need to add TEMPLATE directive
          }
        } catch {
          // ignore parse errors
        }
      }
    }

    // Add sensible default parameters
    modelfileLines.push(
      `PARAMETER temperature 0.7`,
      `PARAMETER top_p 0.9`,
      ``,
    )

    const modelfilePath = path.join(outputDir, "Modelfile")
    const modelfileContent = modelfileLines.join("\n")
    fs.writeFileSync(modelfilePath, modelfileContent, "utf-8")

    console.log(`[save] Using ${hasGGUF ? "GGUF" : "safetensors"} format`)
    console.log(`[save] Model path: ${modelPath}`)
    console.log(`[save] Modelfile written to ${modelfilePath}`)
    console.log(`[save] Modelfile content:\n${modelfileContent}`)
    console.log(`[save] Output dir files: ${files.join(", ")}`)

    // Run ollama create
    const ollamaCmd = `ollama create ${modelName} -f "${modelfilePath}"`
    console.log(`[save] Running: ${ollamaCmd}`)
    const output = execSync(ollamaCmd, {
      timeout: 600000, // 10 min for large models
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    })
    console.log(`[save] Ollama output: ${output}`)

    return NextResponse.json({
      success: true,
      modelName,
      message: `Model "${modelName}" created in Ollama successfully.`,
    })
  } catch (error) {
    console.error("Save to Ollama error:", error)
    const errMsg = error instanceof Error ? error.message : "Failed to save model to Ollama"
    return NextResponse.json(
      { error: errMsg },
      { status: 500 }
    )
  }
}
