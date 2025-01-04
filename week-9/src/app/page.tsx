"use client"

import { useState } from "react"
import { Response } from "@prisma/client"
import { PromptForm } from "@/components/prompt-form"
import { ResultCard } from "@/components/result-card"
import { LineChart } from "@tremor/react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function Home() {
  const [results, setResults] = useState<Response[]>([])

  const onSubmit = async (data: { prompt: string }) => {
    try {
      const response = await fetch("/api/prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: data.prompt }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch")
      }
      const result = await response.json()
      setResults((prev) => [...prev, result.data])
    } catch (error) {
      console.error(error)
      throw error
    }
  }

  const chartData = results.map((result, index) => ({
    index: index + 1,
    accuracy: result.accuracy,
    relevancy: result.relevancy,
  }))

  return (
    <div className="container mx-auto p-8">
      <h1 className="text-3xl font-bold mb-8">LLM Evaluation Dashboard</h1>

      <div className="mb-8">
        <PromptForm onSubmit={onSubmit} />
      </div>

      {results.length > 0 && (
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Performance Over Time</CardTitle>
          </CardHeader>
          <CardContent>
            <LineChart
              className="h-80"
              data={chartData}
              index="index"
              categories={["accuracy", "relevancy"]}
              colors={["blue", "green"]}
              yAxisWidth={40}
            />
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {results.map((result, index) => (
          <ResultCard key={index} result={result} />
        ))}
      </div>
    </div>
  )
}

