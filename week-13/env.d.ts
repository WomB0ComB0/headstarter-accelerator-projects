declare global {
  namespace NodeJS {
    interface ProcessEnv {
      GOOGLE_AI_API_KEY: string;
    }
  }
}

export { };