import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Response } from "@prisma/client"

interface ResultCardProps {
  result: Response
}

export function ResultCard({ result }: ResultCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Model: Gemini 1.5</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <p>Accuracy: {result.accuracy?.toFixed(2)}</p>
          <p>Relevancy: {result.relevancy?.toFixed(2)}</p>
          <p className="mt-4 text-sm text-muted-foreground">{result.response}</p>
        </div>
      </CardContent>
    </Card>
  )
}

