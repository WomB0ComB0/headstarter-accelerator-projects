import { ClerkProvider } from "@clerk/nextjs";
import { UploadThingProvider } from "./providers/uploadthing-provider";
import { Navigation } from "./components/navigation";
import { Toaster } from "sonner";
import "./globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <title>TikTok Clone</title>
        <meta name="description" content="Share and discover short videos" />
      </head>
      <ClerkProvider>
        <body className="min-h-screen bg-black text-white">
          <div className="flex">
            <Navigation />
            <main className="flex-1 ml-[240px]">
              <UploadThingProvider>
                {children}
              </UploadThingProvider>
            </main>
          </div>
          <Toaster />
        </body>
      </ClerkProvider>
    </html>
  );
} 