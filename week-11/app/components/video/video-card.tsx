"use client";

import { useState, useRef } from 'react';
import { Video } from '~/app/types';
import { useUser } from '@clerk/nextjs';
import { Heart, MessageCircle, Share2 } from 'lucide-react';
import { formatViewCount } from '~/app/lib/utils';

interface VideoCardProps {
  video: Video;
}

export function VideoCard({ video }: VideoCardProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const { isSignedIn } = useUser();

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <div className="relative h-[calc(100vh-80px)] snap-start">
      <video
        ref={videoRef}
        src={video.url}
        className="w-full h-full object-cover"
        loop
        onClick={togglePlay}
      />
      
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/60 to-transparent">
        <h3 className="font-semibold text-lg">{video.title}</h3>
        <p className="text-sm opacity-90">{video.description}</p>
      </div>

      <div className="absolute right-4 bottom-20 flex flex-col gap-4">
        <button className="p-2 bg-gray-800/60 rounded-full" onClick={() => {}}>
          <Heart className={`w-6 h-6 ${video.likes > 0 ? 'text-red-500' : ''}`} />
          <span className="text-sm">{formatViewCount(video.likes)}</span>
        </button>
        <button className="p-2 bg-gray-800/60 rounded-full">
          <MessageCircle className="w-6 h-6" />
        </button>
        <button className="p-2 bg-gray-800/60 rounded-full">
          <Share2 className="w-6 h-6" />
        </button>
      </div>
    </div>
  );
} 