import { Groq } from "groq-sdk";
import { env } from "../env";
import { ChatCompletionCreateParamsBase } from "groq-sdk/resources/chat/completions";

/**
 * Initialized Groq client instance for making API calls to Groq's language models.
 * Uses the GROQ_API_KEY from environment variables for authentication.
 * @see {@link https://groq.com/docs} for Groq API documentation
 */
export const groq = new Groq({
  apiKey: env.GROQ_API_KEY,
});

/**
 * Type definition for supported Groq chat model identifiers.
 * Extracted from the base chat completion parameters to ensure type safety.
 * @typedef {ChatCompletionCreateParamsBase["model"]} GroqChatModel
 */
export type GroqChatModel = ChatCompletionCreateParamsBase["model"];

/**
 * The default Groq model used for chat completions.
 * Currently set to Mixtral 8x7B with 32k context window.
 * 
 * Mixtral 8x7B is a mixture-of-experts model that offers:
 * - Strong performance across various tasks
 * - 32k token context window
 * - Good balance of speed and quality
 * 
 * @constant
 * @type {GroqChatModel}
 */
export const GROQ_MODEL: GroqChatModel = "mixtral-8x7b-32768";
