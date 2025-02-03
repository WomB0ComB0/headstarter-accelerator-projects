import { NextResponse } from 'next/server';
import { z } from 'zod';
import { supabase } from '~/app/lib/supabase';
import { VIDEO_CATEGORIES } from '~/app/config/constants';

const analyzeSchema = z.object({
  videoUrl: z.string().url(),
  userId: z.string().uuid(),
  metadata: z.object({
    title: z.string().min(1).max(100),
    description: z.string().max(500).optional(),
    category: z.enum(VIDEO_CATEGORIES),
    duration: z.number().min(0)
  })
});

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { videoUrl, userId, metadata } = analyzeSchema.parse(body);

    const { data, error } = await supabase
      .from('videos')
      .insert({
        url: videoUrl,
        user_id: userId,
        title: metadata.title,
        description: metadata.description || '',
        category: metadata.category,
        duration: metadata.duration,
        is_moderated: true,
        views: 0,
        likes: 0,
        thumbnail_url: null
      })
      .select('*')
      .single();

    if (error) {
      console.error('Supabase error:', error);
      return NextResponse.json(
        { error: 'Database error' },
        { status: 500 }
      );
    }

    return NextResponse.json({ success: true, video: data });
  } catch (error) {
    console.error('Error analyzing video:', error);
    return NextResponse.json(
      { error: 'Failed to analyze video' },
      { status: 500 }
    );
  }
}
