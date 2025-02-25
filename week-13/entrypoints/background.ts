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
    browser.runtime.onMessage.addListener(async (message) => {
      console.log('[Background] Received message:', message);
      if (message.type === 'test') {
        return 'Background connected';
      }
      // Handle text generation requests
      if (message.type === 'getCompletion') {
        try {
          // Initialize Gemini model with configured settings
          const geminiModel = genAI.getGenerativeModel({ 
            model,
            generationConfig,
            safetySettings
          });

          console.log('[Background] Requesting completion for:', message.prompt);
          const result = await geminiModel.generateContent(message.prompt);
          const completion = await result.response.text();
          console.log('[Background] Got completion:', completion);
          
          return completion;
        } catch (error) {
          console.error('[Background] Gemini API error:', error);
          return '';
        }
      }
    });
  }
} satisfies BackgroundDefinition);
