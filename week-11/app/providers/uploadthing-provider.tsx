"use client";

import { UploadDropzone } from "@uploadthing/react";
import { generateComponents } from "@uploadthing/react";
import type { OurFileRouter } from "~/app/utils/uploadthing";

export const { useUploadThing } = generateComponents<OurFileRouter>();

export function UploadThingProvider({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
} 