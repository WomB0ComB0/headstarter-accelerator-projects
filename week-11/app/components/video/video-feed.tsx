"use client";

import { useEffect, useState } from 'react';
import { useInView } from 'react-intersection-observer';
import { supabase } from '~/app/lib/supabase';
import { Video } from '~/app/types';
import { VideoCard } from './video-card';

export function VideoFeed() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { ref, inView } = useInView();

  async function getVideos(pageNum: number): Promise<Video[]> {
    try {
      const { data, error } = await supabase
        .from('videos')
        .select('*')
        .eq('is_moderated', true)
        .order('created_at', { ascending: false })
        .range((pageNum - 1) * 10, pageNum * 10 - 1);

      if (error) throw error;
      return data || [];
    } catch (err) {
      setError('Failed to load videos');
      return [];
    }
  }

  useEffect(() => {
    if (inView && !loading) {
      setLoading(true);
      getVideos(page)
        .then((newVideos) => {
          setVideos((prev) => [...prev, ...newVideos]);
          setPage((p) => p + 1);
        })
        .catch((err) => setError('Failed to load videos'))
        .finally(() => setLoading(false));
    }
  }, [inView, page]);

  if (error) {
    return (
      <div className="flex justify-center items-center h-screen">
        <p className="text-red-500">{error}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {videos.length === 0 && !loading ? (
        <div className="flex justify-center items-center h-screen">
          <p className="text-gray-500">No videos found</p>
        </div>
      ) : (
        <>
          {videos.map((video) => (
            <VideoCard key={video.id} video={video} />
          ))}
          {loading && (
            <div className="flex justify-center p-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
            </div>
          )}
          <div ref={ref} className="h-10" />
        </>
      )}
    </div>
  );
}
