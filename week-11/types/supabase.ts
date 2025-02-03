export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      videos: {
        Row: {
          id: string
          title: string
          description: string | null
          url: string
          thumbnail_url: string | null
          user_id: string
          created_at: string
          views: number
          likes: number
          duration: number
          is_moderated: boolean
          category: string
        }
        Insert: {
          id?: string
          title: string
          description?: string | null
          url: string
          thumbnail_url?: string | null
          user_id: string
          created_at?: string
          views?: number
          likes?: number
          duration: number
          is_moderated?: boolean
          category: string
        }
        Update: {
          id?: string
          title?: string
          description?: string | null
          url?: string
          thumbnail_url?: string | null
          user_id?: string
          created_at?: string
          views?: number
          likes?: number
          duration?: number
          is_moderated?: boolean
          category?: string
        }
      }
      // Add other tables here
    }
  }
} 