import { NextResponse } from "next/server";
import { db as prisma } from "@/lib/prisma";
import { evaluateResponse } from "@/lib/evaluation";
import { genAI } from "@/lib/gemini-settings";

export async function POST(req: Request) {
  try {
    const { prompt } = await req.json();
    
    // Create prompt record
    const promptRecord = await prisma.prompt.create({
      data: { text: prompt }
    });

    // Get response from Gemini
    const startTime = Date.now();
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    const geminiResponse = await model.generateContent(prompt);
    const responseText = geminiResponse.response.text();
    const responseTime = Date.now() - startTime;

    // Evaluate response
    const evaluation = await evaluateResponse(prompt, responseText);

    // Store response with metrics
    const response = await prisma.response.create({
      data: {
        promptId: promptRecord.id,
        llmProvider: "gemini-pro",
        response: responseText,
        accuracy: evaluation.accuracy,
        relevancy: evaluation.relevancy,
        responseTime
      }
    });

    return NextResponse.json({ success: true, data: response });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ success: false, error: "Internal Server Error" }, { status: 500 });
  }
} 