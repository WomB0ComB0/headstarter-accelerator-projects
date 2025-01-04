-- CreateTable
CREATE TABLE "Prompt" (
    "id" TEXT NOT NULL,
    "text" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Prompt_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Response" (
    "id" TEXT NOT NULL,
    "promptId" TEXT NOT NULL,
    "llmProvider" TEXT NOT NULL,
    "response" TEXT NOT NULL,
    "accuracy" DOUBLE PRECISION,
    "relevancy" DOUBLE PRECISION,
    "responseTime" DOUBLE PRECISION NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Response_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "Response" ADD CONSTRAINT "Response_promptId_fkey" FOREIGN KEY ("promptId") REFERENCES "Prompt"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
