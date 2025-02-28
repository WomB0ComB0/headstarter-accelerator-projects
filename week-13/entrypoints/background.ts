import type { BackgroundDefinition } from 'wxt';
/**
 * Configuration settings for Google's Gemini AI model integration
 * @module gemini-settings
 */

import { GoogleGenerativeAI, HarmBlockThreshold, HarmCategory } from '@google/generative-ai';
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config({ path: '.env' });

/**
 * The specific Gemini model version to use
 * gemini-1.5-flash is optimized for faster response times
 */
export const model = 'gemini-2.0-flash';

/**
 * Google AI API key retrieved from environment variables
 * @throws {Error} If GOOGLE_AI_API_KEY environment variable is not set
 */
export const API_KEY = process.env.GOOGLE_AI_API_KEY || (() => {
  throw new Error('GOOGLE_AI_API_KEY is not set');
})();

/**
 * Initialized instance of Google's Generative AI client
 * Used to interact with the Gemini API
 */
export const genAI = new GoogleGenerativeAI(API_KEY);

/**
 * Generation configuration settings for the model
 * Currently undefined to use model defaults
 * Can be configured with parameters like:
 * - temperature
 * - topK
 * - topP
 * - maxOutputTokens
 */
export const generationConfig = undefined;

/**
 * Safety settings that control content filtering for generated responses
 * All safety filters are currently set to BLOCK_NONE for maximum permissiveness
 * 
 * @property {Array<Object>} safetySettings - Array of safety threshold configurations
 * @property {HarmCategory} safetySettings[].category - Category of harmful content to filter
 * @property {HarmBlockThreshold} safetySettings[].threshold - Threshold level for blocking content
 */
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

const systemPrompt = `
You are a sentence completion AI. Your task is to **either autocomplete the last word, complete the last sentence, or add a natural follow-up sentence** based on the given input.

### Instructions:
- The input consists of the full text from a textarea. **Do not modify or repeat the existing text—only generate what comes next**.
- Your response must **either**:
  1. **Complete an unfinished word**, leaving it open-ended if it's part of an incomplete thought or sentence.
  2. **Finish the last sentence** if the current sentence appears complete. **Do not add a period if the sentence still feels open-ended and can continue naturally**. If the sentence is definitely finished, complete it and end with a period.
  3. If the last sentence is complete, generate **one additional coherent sentence** that flows naturally.
- **Maintain proper spacing, punctuation, and formatting**:
  - If the last word is incomplete, complete it without unnecessarily finishing the sentence.
  - If a space is missing before your completion, add it.
  - If the last sentence is already complete, start a new one with proper capitalization.
  - Otherwise, do not modify the existing spacing.
- If the last word is gibberish (e.g., "sdkhbsadhja") and cannot be meaningfully completed, **return an empty string**.
- **Do not generate apologies, explanations, or anything unrelated to continuing the text**.

### Example Behavior:
- Input: "My name is Bob and I li"  
  Output: "ke to cook."  

- Input: "It was a wonderful day at the"  
  Output: " beach."  

- Input: "sdkhbsadhja"  
  Output: ""  

- Input: "I had a good day at the beach."  
  Output: "I played volleyball."  

- Input: "She walked into the room and sat down"  
  Output: " She looked around the room."  

- Input: "I had some challe"  
  Output: "nges I had to overcome." (doesn't end the sentence; leaves it open)

- Input: "I went to the store and bought a new"  
  Output: " pair of shoes." (adds a natural completion without finishing the sentence)
`;

const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${API_KEY}`;

async function callGemini(content: string, context: string) {
  const requestBody = {
    contents: [
      {
        parts: [
          {
            text: `${systemPrompt}.\nHere is website context: ${context}.\nHere is the text from the user: ${content}.`,
          },
        ],
      },
    ],
  };

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const data = await response.json();
    console.log("data from Gemini API:", data);
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (text && text !== '""') {
      return text;
    } else {
      return "";
    }
  } catch (error) {
    console.error("Error calling Gemini API:", error);
    return "";
  }
}

/**
 * Background script for handling AI text generation using Google's Gemini API
 * 
 * This background script sets up a message listener to handle text generation requests
 * from other parts of the extension. It uses the Gemini AI model to generate responses
 * based on provided prompts.
 * 
 * @satisfies BackgroundDefinition - Type checking for WXT background script definition
 */
export default defineBackground({
  main() {
    // Add API key check
    if (!process.env.GOOGLE_AI_API_KEY) {
      console.error('[Background] Missing GOOGLE_AI_API_KEY');
    }

    /**
     * Listens for messages from other extension components
     * 
     * @param message - The message object containing:
     *   - type: String identifying the message type ('getCompletion')
     *   - prompt: String containing the text prompt for generation
     * @returns Promise<string> The generated text response from Gemini
     */
    browser.runtime.onMessage.addListener((message, sender) => {
      if (message.type === "TEXTAREA_UPDATE") {
        console.log("Received Text:", message.value);
        console.log("Received Context:", message.context);

        return callGemini(message.value, message.context || "")
          .then((result) => {
            console.log("generated suggestion:", result);
            if (result) {
              return { success: true, result }; // Send result with response
            }
            return { success: false, error: "No result generated" };
          })
          .catch((error) => {
            console.error("Error in callGemini:", error);
            return { success: false, error: error.message };
          });
      }
      
      return false;
    });
  }
} satisfies BackgroundDefinition);
