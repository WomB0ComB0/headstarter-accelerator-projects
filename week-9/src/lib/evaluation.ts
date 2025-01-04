import { genAI } from "@/lib/gemini-settings";

export async function evaluateResponse(prompt: string, response: string) {
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
  
  const evaluationPrompt = `You are an expert at evaluating LLM responses. Score the following response for accuracy and relevancy on a scale of 0-1.

Prompt: ${prompt}
Response: ${response}

Return only a JSON object in this exact format without any additional text, quotes, or markdown: {"accuracy": 0.X, "relevancy": 0.X}`;

  try {
    const evaluation = await model.generateContent(evaluationPrompt);
    let content = evaluation.response.text();
    
    // More thorough cleanup of the response
    content = content
      .replace(/```[a-z]*\n?/g, '') // Remove ```json or any other language identifier
      .replace(/```/g, '')          // Remove remaining backticks
      .replace(/\n/g, '')           // Remove newlines
      .trim();                      // Remove whitespace
    
    // Extract JSON if it's embedded in other text
    const jsonMatch = content.match(/\{.*\}/);
    if (jsonMatch) {
      content = jsonMatch[0];
    }

    return JSON.parse(content || "{}");
  } catch (error) {
    console.error("Failed to parse evaluation:", error);
    return { accuracy: 0.5, relevancy: 0.5 };
  }
}