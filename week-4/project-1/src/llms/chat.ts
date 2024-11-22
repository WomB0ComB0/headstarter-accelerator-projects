import { ChatCompletionCreateParamsNonStreaming } from "groq-sdk/resources/chat/completions";
import { groq, GROQ_MODEL } from "./groq";

/**
 * Generates a chat completion response using the Groq API.
 * 
 * This function creates a chat completion by sending a request to the Groq API with the specified options.
 * It uses a fixed model (GROQ_MODEL) and temperature (0) for consistent, deterministic responses.
 * 
 * @async
 * @param {Omit<ChatCompletionCreateParamsNonStreaming, "model">} options - Configuration options for the chat completion.
 *   Accepts all ChatCompletionCreateParamsNonStreaming parameters except 'model' which is fixed.
 *   Common options include:
 *   - messages: Array of chat messages to generate completion from
 *   - max_tokens: Maximum number of tokens to generate
 *   - top_p: Nucleus sampling threshold
 *   - frequency_penalty: Penalty for using frequent tokens
 *   - presence_penalty: Penalty for using new tokens
 * 
 * @returns {Promise<Object>} The generated message from the first choice in the completion response.
 *   Contains properties like:
 *   - role: The role of the message (usually 'assistant')
 *   - content: The actual text content of the generated response
 * 
 * @throws {Error} If the API request fails or returns an invalid response
 * 
 * @example
 * const response = await generateChatCompletion({
 *   messages: [
 *     { role: "user", content: "Hello, how are you?" }
 *   ]
 * });
 * console.log(response.content); // Generated response text
 */
export const generateChatCompletion = async (
  options: Omit<ChatCompletionCreateParamsNonStreaming, "model">
) => {
  const response = await groq.chat.completions.create({
    model: GROQ_MODEL,
    temperature: 0,
    ...options,
  });
  return response.choices[0].message;
};
