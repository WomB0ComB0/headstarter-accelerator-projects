import { GoogleGenerativeAI, HarmBlockThreshold, HarmCategory } from '@google/generative-ai';

export const model = 'gemini-1.5-flash';
export const API_KEY = process.env.NEXT_PUBLIC_GOOGLE_AI_API_KEY || (() => { throw new Error('NEXT_PUBLIC_GOOGLE_AI_API_KEY is not set') })();
export const genAI = new GoogleGenerativeAI(API_KEY);

export const generationConfig = undefined;
export const safetySettings = [
  {
    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
];
