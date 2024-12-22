'use server';

import { config } from '@/config/config';
import * as Sentry from '@sentry/nextjs';
import { registerOTel } from '@vercel/otel';

/**
 * Registers and configures instrumentation services for the application.
 * This includes OpenTelemetry for observability and Sentry for error tracking.
 *
 * @async
 * @function register
 * @returns {Promise<void>} A promise that resolves when all instrumentation is configured
 *
 * @example
 * ```ts
 * // This is typically called automatically by Next.js
 * await register();
 * ```
 *
 * @remarks
 * - Initializes OpenTelemetry with the service name 'ktp-national'
 * - Conditionally imports Sentry configs based on runtime environment
 * - Handles both Node.js and Edge runtime environments
 * - Should be called before the application starts handling requests
 *
 * @throws {Error} May throw if Sentry config imports fail or if OTel registration fails
 */
export async function register() {
  const runtime = process.env.NEXT_RUNTIME;

  try {
    if (process.env.NEXT_PUBLIC_VERCEL_ENV) {
      registerOTel({
        serviceName: 'pentagram',
        instrumentations: runtime === 'edge' ? [] : undefined,
      });
    }

    if (config.sentry.dsn) {
      if (runtime === 'edge') {
        await import('../sentry.edge.config');
      } else if (runtime === 'nodejs') {
        await import('../sentry.server.config');
      }
    }
  } catch (error) {
    console.error(`[${runtime} Instrumentation] Initialization error:`, error);
  }
}

/**
 * Error handler function that captures and reports request errors to Sentry.
 * Direct export of Sentry's captureRequestError function for use in error boundaries
 * or request handlers.
 *
 * @function onRequestError
 * @type {typeof Sentry.captureRequestError}
 *
 * @example
 * ```ts
 * try {
 *   await handleRequest(req);
 * } catch (error) {
 *   onRequestError(error);
 * }
 * ```
 *
 * @remarks
 * - Automatically captures request context and error details
 * - Integrates with Sentry's error tracking system
 * - Preserves error stack traces and metadata
 * - Should be used for handling request-specific errors
 */
export const onRequestError = Sentry.captureException;
