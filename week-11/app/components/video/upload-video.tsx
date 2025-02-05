"use client";

import { useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { useUser } from '@clerk/nextjs';
import { UploadDropzone } from '@uploadthing/react';
import { toast } from 'sonner';
import { 
  AlignLeft,
  CloudUpload,
  Hash,
  MessageSquare,
  Loader2,
  X
} from 'lucide-react';
import type { OurFileRouter } from "~/app/utils/uploadthing";
import { VIDEO_CATEGORIES } from '~/app/config/constants';

type VideoCategory = typeof VIDEO_CATEGORIES[number];

export function UploadVideo() {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState<VideoCategory>(VIDEO_CATEGORIES[0]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const { user } = useUser();
  const router = useRouter();

  const handleVideoLoad = () => {
    if (videoRef.current) {
      setDuration(Math.round(videoRef.current.duration));
    }
  };

  const handlePublish = async () => {
    if (!previewUrl || !title) return;
    
    try {
      setUploading(true);
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          videoUrl: previewUrl,
          userId: user?.id,
          metadata: {
            title: title || 'Untitled',
            description,
            category,
            duration: duration || 0
          }
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to process video');
      }

      toast.success('Video published successfully!', {
        icon: '🎉',
      });
      router.push('/');
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Failed to publish video', {
        icon: '❌',
      });
    } finally {
      setUploading(false);
    }
  };

  if (!user) {
    return (
      <div className="flex flex-col items-center justify-center h-[80vh] text-center">
        <CloudUpload className="w-16 h-16 mb-4 text-gray-400" />
        <h2 className="mb-2 text-xl font-semibold">Sign in to upload videos</h2>
        <p className="text-gray-400">Join our community to start sharing your videos</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl p-6 mx-auto">
      <div className="flex flex-col gap-8 md:flex-row">
        {/* Left side - Upload and Preview */}
        <div className="flex-1">
          <div className="p-6 mb-4 bg-gray-900 rounded-xl">
            <h2 className="flex items-center mb-4 text-xl font-semibold">
              <CloudUpload className="w-5 h-5 mr-2" />
              Upload Video
            </h2>
            
            {!previewUrl ? (
              <div className="relative">
                <UploadDropzone<OurFileRouter, "videoUploader">
                  endpoint="videoUploader"
                  onUploadBegin={() => {
                    setUploading(true);
                    setUploadProgress(0);
                    toast.info('Upload started...', {
                      icon: '🎥',
                    });
                  }}
                  onUploadProgress={(progress) => {
                    setUploadProgress(Math.round(progress));
                  }}
                  onClientUploadComplete={async (res) => {
                    if (!res?.[0]) return;
                    setPreviewUrl(res[0].url);
                    setUploading(false);
                    setUploadProgress(100);
                    toast.success('Video uploaded! Add some details.', {
                      icon: '✨',
                    });
                  }}
                  onUploadError={(error: Error) => {
                    console.error('Upload error:', error);
                    toast.error(error.message || 'Upload failed', {
                      icon: '❌',
                    });
                    setUploading(false);
                    setUploadProgress(0);
                  }}
                  className="ut-uploading:opacity-50 ut-uploading:cursor-not-allowed
                            border-2 border-dashed border-gray-700 rounded-xl
                            ut-label:text-gray-300 ut-button:bg-blue-600 
                            ut-button:hover:bg-blue-700 min-h-[300px]
                            flex flex-col items-center justify-center"
                />
                
                {uploading && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm rounded-xl">
                    {/* Progress bar */}
                    <div className="w-64 h-4 mb-4 bg-gray-700 rounded-full">
                      <div 
                        className="h-4 transition-all duration-300 bg-blue-500 rounded-full"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    <p className="mb-4 text-lg font-medium">{uploadProgress}% uploaded</p>
                    <button
                      onClick={() => {
                        setUploading(false);
                        setUploadProgress(0);
                        toast.error('Upload cancelled');
                      }}
                      className="flex items-center gap-2 px-4 py-2 transition-colors border border-red-500 rounded-lg bg-red-500/20 hover:bg-red-500/30"
                    >
                      <X className="w-4 h-4" />
                      Cancel Upload
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="relative overflow-hidden rounded-lg aspect-video">
                <video
                  ref={videoRef}
                  src={previewUrl}
                  className="object-cover w-full h-full"
                  controls
                  onLoadedMetadata={handleVideoLoad}
                />
              </div>
            )}
          </div>
        </div>

        {/* Right side - Video Details */}
        <div className="flex-1">
          <div className="p-6 bg-gray-900 rounded-xl">
            <div className="space-y-6">
              <div>
                <label className="flex items-center mb-2 text-sm font-medium">
                  <AlignLeft className="w-4 h-4 mr-2" />
                  Title
                </label>
                <input
                  type="text"
                  placeholder="Add a title that describes your video"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="w-full p-3 transition-all bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  maxLength={100}
                />
                <p className="mt-1 text-xs text-gray-400">
                  {title.length}/100 characters
                </p>
              </div>

              <div>
                <label className="flex items-center mb-2 text-sm font-medium">
                  <Hash className="w-4 h-4 mr-2" />
                  Category
                </label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value as VideoCategory)}
                  className="w-full p-3 transition-all bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {VIDEO_CATEGORIES.map((cat) => (
                    <option key={cat} value={cat}>{cat}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="flex items-center mb-2 text-sm font-medium">
                  <MessageSquare className="w-4 h-4 mr-2" />
                  Description
                </label>
                <textarea
                  placeholder="Tell viewers about your video"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="w-full p-3 bg-gray-800 rounded-lg border border-gray-700 
                           focus:ring-2 focus:ring-blue-500 focus:border-transparent
                           transition-all min-h-[100px]"
                  maxLength={500}
                />
                <p className="mt-1 text-xs text-gray-400">
                  {description.length}/500 characters
                </p>
              </div>

              <button
                onClick={handlePublish}
                disabled={!previewUrl || uploading || !title}
                className={`w-full py-3 px-4 rounded-lg font-medium
                          ${!previewUrl || uploading || !title
                            ? 'bg-gray-700 cursor-not-allowed'
                            : 'bg-blue-600 hover:bg-blue-700'} 
                          transition-colors flex items-center justify-center gap-2`}
              >
                {uploading && <Loader2 className="w-4 h-4 animate-spin" />}
                {uploading ? 'Publishing...' : 'Publish Video'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
