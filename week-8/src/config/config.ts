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
});

const loadConfig = (): Config => {
  try {
    const config: Config = {
      app: {
        environment: process.env.NODE_ENV as 'development' | 'production' | 'test',
      },
      integration: {
        redis: {
          url:
            process.env.UPSTASH_REDIS_REST_URL ||
            (() => {
              throw new Error('UPSTASH_REDIS_REST_URL is not set');
            })(),
          token:
            process.env.UPSTASH_REDIS_REST_TOKEN ||
            (() => {
              throw new Error('UPSTASH_REDIS_REST_TOKEN is not set');
            })(),
        },
      },
    };

    const parsedConfig = configSchema.parse(config);
    return parsedConfig;
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error('Failed to load configuration', error.errors);
      throw new Error('Failed to load configuration');
    }
    throw error;
  }
};

const config = loadConfig();

function getConfig<K extends keyof Config>(
  category: K,
): Config[K] {
  return config[category];
}

export { getConfig, config };