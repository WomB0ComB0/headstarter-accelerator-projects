export * from './supabase';
export * from './video';
export * from './user';

export interface Video {
  id: string;
  title: string;
  description?: string;
  url: string;
  thumbnail_url?: string;
  user_id: string;
  created_at: string;
  views: number;
  likes: number;
  duration: number;
  is_moderated: boolean;
  category: string;
}

export interface Comment {
  id: string;
  video_id: string;
  user_id: string;
  content: string;
  created_at: string;
  likes: number;
}

export interface UserProfile {
  id: string;
  username: string;
  full_name: string;
  avatar_url: string;
  bio?: string;
  following_count: number;
  followers_count: number;
  interests: string[];
}

export interface VideoEngagement {
  user_id: string;
  video_id: string;
  action_type: 'view' | 'like' | 'share' | 'favorite';
  created_at: string;
} 