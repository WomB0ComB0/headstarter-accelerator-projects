export const ROUTES = {
  HOME: '/',
  PROFILE: '/profile',
  UPLOAD: '/upload',
  VIDEO: '/video',
  FOLLOWING: '/following',
  TRENDING: '/trending'
} as const;

export const VIDEO_CATEGORIES = [
  'Comedy',
  'Music',
  'Dance',
  'Food',
  'Fashion',
  'Sports',
  'Education',
  'Technology',
  'Travel',
  'Pets'
] as const;

export const MODERATION_RULES = {
  MAX_VIDEO_DURATION: 180, // 3 minutes
  MIN_VIDEO_DURATION: 3,   // 3 seconds
  MAX_TITLE_LENGTH: 100,
  MAX_DESCRIPTION_LENGTH: 500,
  ALLOWED_VIDEO_TYPES: ['video/mp4', 'video/quicktime'],
  MAX_FILE_SIZE: `${100 * 1024 * 1024}mb` // 100MB
} as const;

export const FEED_BATCH_SIZE = 10; 