import { genAI } from "@/lib/gemini-settings";

export async function evaluateResponse(prompt: string, response: string) {
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
  
  const evaluationPrompt = `You are an expert at evaluating LLM responses. Score the following response for accuracy and relevancy on a scale of 0-1.

Prompt: ${prompt}
Response: ${response}

Provide scores in JSON format: {"accuracy": number, "relevancy": number}`;

  try {
    const evaluation = await model.generateContent(evaluationPrompt);
    const content = evaluation.response.text();
    return JSON.parse(content || "{}");
  } catch (error) {
    console.error("Failed to parse evaluation:", error);
    return { accuracy: 0.5, relevancy: 0.5 };
  }
} 