/**
 * Environment variable configuration and validation module
 * @module env
 */

import * as dotenv from "dotenv";
import { createPrivateKey } from "crypto";
import chalk from "chalk";

// Load environment variables from .env file
dotenv.config();

/**
 * Environment variable configuration object
 * @constant
 * @type {Object}
 * @property {string} GITHUB_APP_ID - GitHub App ID for authentication
 * @property {string} GITHUB_PRIVATE_KEY - Private key for GitHub App authentication
 * @property {string} GITHUB_WEBHOOK_SECRET - Secret for validating GitHub webhooks
 * @property {string} GROQ_API_KEY - API key for GROQ services
 */
export const env = {
  GITHUB_APP_ID: process.env.GITHUB_APP_ID,
  GITHUB_PRIVATE_KEY: process.env.GITHUB_PRIVATE_KEY,
  GITHUB_WEBHOOK_SECRET: process.env.GITHUB_WEBHOOK_SECRET,
  GROQ_API_KEY: process.env.GROQ_API_KEY,
} as const;

/**
 * Tracks validation state of environment variables
 * @type {boolean}
 */
let valid = true;

/**
 * Validates presence of all required environment variables
 * Logs error messages for any missing variables
 */
for (const key in env) {
  if (!env[key as keyof typeof env]) {
    console.log(
      chalk.red("✖") +
        chalk.gray(" Missing required env var: ") +
        chalk.bold(`process.env.${key}`)
    );
    valid = false;
  }
}

/**
 * Validates format of GitHub private key
 * Attempts to parse the key and logs detailed error if invalid
 * Key must be in RSA format with proper header/footer
 */
try {
  createPrivateKey(env.GITHUB_PRIVATE_KEY);
} catch (error) {
  console.log(
    chalk.red(
      "\n✖ Invalid GitHub private key format for " +
        chalk.bold(`process.env.GITHUB_PRIVATE_KEY`) +
        "\n"
    ) +
      chalk.gray("  • Must start with: ") +
      chalk.bold("-----BEGIN RSA PRIVATE KEY-----\n") +
      chalk.gray("  • Must end with:   ") +
      chalk.bold("-----END RSA PRIVATE KEY-----\n")
  );
  valid = false;
}

/**
 * Exits process if any validation checks failed
 * Displays final error message prompting user to check .env file
 */
if (!valid) {
  console.log(
    chalk.yellow("\n⚠ ") +
      chalk.bold("Please check your .env file and try again.\n")
  );
  process.exit(1);
}
