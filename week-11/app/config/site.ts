export const siteConfig = {
  name: "VideoShare",
  description: "Share and discover short-form videos",
  url: process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
  ogImage: "/og.png",
  links: {
    github: "https://github.com/yourusername/video-share",
    twitter: "https://twitter.com/yourusername"
  },
  creator: "Your Name"
} as const;

export type SiteConfig = typeof siteConfig; 