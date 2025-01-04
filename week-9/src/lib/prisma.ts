/**
 * @fileoverview Prisma database client configuration and initialization
 * @module lib/prisma
 */

import { PrismaClient } from '@prisma/client';

/**
 * Extends the global namespace to include prisma client type
 * This allows for global singleton pattern implementation
 */
declare global {
  var prisma: PrismaClient | undefined;
}

let prisma: PrismaClient;

if (process.env.NODE_ENV === 'production') {
  prisma = new PrismaClient({
    log: ['error'],
  });
} else {
  if (!global.prisma) {
    global.prisma = new PrismaClient({
      log: ['query', 'error', 'warn'],
    });
  }
  prisma = global.prisma;
}

/**
 * Database client singleton instance
 *
 * @constant
 * @type {PrismaClient}
 *
 * @remarks
 * Uses global singleton pattern to prevent multiple client instances:
 * - Reuses existing global client if available
 * - Creates new client if none exists
 *
 * @example
 * import { db } from '@/lib/prisma'
 *
 * // Use db for database operations
 * const user = await db.user.findUnique({
 *   where: { id: 1 }
 * })
 */
export const db = prisma;

/**
 * Re-export PrismaClient type for use in other modules
 */
export type { PrismaClient };