"use client"

import { useState } from "react"
import { useForm } from "react-hook-form"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ReloadIcon } from "@radix-ui/react-icons"

interface PromptFormProps {
  onSubmit: (data: { prompt: string }) => Promise<void>
}

export function PromptForm({ onSubmit }: PromptFormProps) {
  const { register, handleSubmit, formState: { isSubmitting } } = useForm<{ prompt: string }>()
  const [error, setError] = useState<string | null>(null)

  const onSubmitWrapper = async (data: { prompt: string }) => {
    setError(null)
    try {
      await onSubmit(data)
    } catch (err) {
      setError("An error occurred while processing your request.")
    }
  }

  return (
    <form onSubmit={handleSubmit(onSubmitWrapper)} className="space-y-4">
      <Textarea
        {...register("prompt", { required: "Prompt is required" })}
        placeholder="Enter your prompt here..."
        rows={4}
      />
      {error && <p className="text-red-500 text-sm">{error}</p>}
      <Button type="submit" disabled={isSubmitting}>
        {isSubmitting && <ReloadIcon className="mr-2 h-4 w-4 animate-spin" />}
        {isSubmitting ? "Processing..." : "Submit"}
      </Button>
    </form>
  )
}

