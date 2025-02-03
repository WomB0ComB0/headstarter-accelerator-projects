import { createUploadthing, type FileRouter } from "uploadthing/next";
import { getAuth } from "@clerk/nextjs/server";
import { MODERATION_RULES } from "~/app/config/constants";

const f = createUploadthing();

export const ourFileRouter = {
  videoUploader: f({
    video: {
      maxFileSize: MODERATION_RULES.MAX_FILE_SIZE,
      maxFileCount: 1
    }
  })
    .middleware(async ({ req }) => {
      const { userId } = await getAuth(req);
      if (!userId) throw new Error("Unauthorized");
      return { userId };
    })
    .onUploadComplete(async ({ metadata, file }) => {
      try {
        console.log("Upload complete for userId:", metadata.userId);
        console.log("File:", file);
        return { uploadedBy: metadata.userId, url: file.url };
      } catch (error) {
        console.error("Upload error:", error);
        throw error;
      }
    })
} satisfies FileRouter;

export type OurFileRouter = typeof ourFileRouter;