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
        <h2 className="text-xl font-semibold mb-2">Sign in to upload videos</h2>
        <p className="text-gray-400">Join our community to start sharing your videos</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="flex flex-col md:flex-row gap-8">
        {/* Left side - Upload and Preview */}
        <div className="flex-1">
          <div className="bg-gray-900 rounded-xl p-6 mb-4">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
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
                  <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center rounded-xl">
                    {/* Progress bar */}
                    <div className="w-64 bg-gray-700 rounded-full h-4 mb-4">
                      <div 
                        className="bg-blue-500 h-4 rounded-full transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    <p className="text-lg font-medium mb-4">{uploadProgress}% uploaded</p>
                    <button
                      onClick={() => {
                        setUploading(false);
                        setUploadProgress(0);
                        toast.error('Upload cancelled');
                      }}
                      className="flex items-center gap-2 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 
                               border border-red-500 rounded-lg transition-colors"
                    >
                      <X className="w-4 h-4" />
                      Cancel Upload
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="relative aspect-video rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  src={previewUrl}
                  className="w-full h-full object-cover"
                  controls
                  onLoadedMetadata={handleVideoLoad}
                />
              </div>
            )}
          </div>
        </div>

        {/* Right side - Video Details */}
        <div className="flex-1">
          <div className="bg-gray-900 rounded-xl p-6">
            <div className="space-y-6">
              <div>
                <label className="flex items-center text-sm font-medium mb-2">
                  <AlignLeft className="w-4 h-4 mr-2" />
                  Title
                </label>
                <input
                  type="text"
                  placeholder="Add a title that describes your video"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="w-full p-3 bg-gray-800 rounded-lg border border-gray-700 
                           focus:ring-2 focus:ring-blue-500 focus:border-transparent
                           transition-all"
                  maxLength={100}
                />
                <p className="text-xs text-gray-400 mt-1">
                  {title.length}/100 characters
                </p>
              </div>

              <div>
                <label className="flex items-center text-sm font-medium mb-2">
                  <Hash className="w-4 h-4 mr-2" />
                  Category
                </label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value as VideoCategory)}
                  className="w-full p-3 bg-gray-800 rounded-lg border border-gray-700 
                           focus:ring-2 focus:ring-blue-500 focus:border-transparent
                           transition-all"
                >
                  {VIDEO_CATEGORIES.map((cat) => (
                    <option key={cat} value={cat}>{cat}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="flex items-center text-sm font-medium mb-2">
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
                <p className="text-xs text-gray-400 mt-1">
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
