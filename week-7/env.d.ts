declare global {
  namespace NodeJS {
    interface ProcessEnv {
      UPSTASH_REDIS_REST_URL: string;
      UPSTASH_REDIS_REST_TOKEN: string;
      GROQ_API_KEY: string;
    }
  }
}

export {};