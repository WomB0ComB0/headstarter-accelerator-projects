import { loadEnvConfig } from '@next/env';

// Load environment variables at the start
const projectDir = process.cwd();
loadEnvConfig(projectDir);

import { z } from 'zod';

interface Config {
  app: {
    environment: 'development' | 'production' | 'test';
  };
  integration: {
    redis: {
      url: string;
      token: string;
    };
  };
  blob: {
    token: string;
  };
  sentry: {
    dsn: string;
    authToken: string;
    suppressTurbo: number;
  };
}

const configSchema = z.object({
  app: z.object({
    environment: z.enum(['development', 'production', 'test']),
  }),
  integration: z.object({
    redis: z.object({
      url: z.string().url().min(1, 'Redis URL is required'),
      token: z.string().min(1, 'Redis token is required'),
    }),
  }),
  blob: z.object({
    token: z.string().min(1, 'Blob token is required'),
  }),
  sentry: z.object({
    dsn: z.string().url().min(1, 'Sentry DSN is required'),
    authToken: z.string().min(1, 'Sentry auth token is required'),
    suppressTurbo: z.number().min(1, 'Sentry suppress turbo pack is required'),
  }),
});

const loadConfig = (): Config => {
  try {
    const config: Config = {
      app: {
        environment: (process.env.NODE_ENV || 'development') as 'development' | 'production' | 'test',
      },
      integration: {
        redis: {
          url: process.env.UPSTASH_REDIS_REST_URL || '',
          token: process.env.UPSTASH_REDIS_REST_TOKEN || '',
        },
      },
      blob: {
        token: process.env.BLOB_READ_WRITE_TOKEN || '',
      },
      sentry: {
        dsn: process.env.SENTRY_DSN || '',
        authToken: process.env.SENTRY_AUTH_TOKEN || '',
        suppressTurbo: Number(process.env.SENTRY_SUPPRESS_TURBOPACK_WARNING || '1'),
      },
    };

    const parsedConfig = configSchema.parse(config);
    return parsedConfig;
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error('Configuration validation failed:', error.errors);
      throw new Error('Configuration validation failed');
    }
    throw error;
  }
};

const config = loadConfig();

export { config };
