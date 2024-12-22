'use client';

import { Card, CardContent } from '@/components/ui/card';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

interface Recommendation {
  recommendations: string[];
}

export default function Home() {
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);


  const fetchRecommendations = async () => {
    try {
      const response = await fetch('/api/generate-image', {
        method: 'GET',
      });
      const data: Recommendation = await response.json();
      if (data.recommendations) {
        setRecommendations(data.recommendations);
      }
    } catch (error) {
      console.error('Failed to fetch recommendations:', error);
    }
  };

  useEffect(() => {
    fetchRecommendations();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/generate-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate image');
      }

      if (data.success && data.image) {
        setGeneratedImage(`data:image/png;base64,${data.image}`);
        if (data.recommendations) {
          setRecommendations(data.recommendations);
        }
      }

      setInputText('');
    } catch (error) {
      setError(error instanceof Error ? error.message : 'An error occurred');
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRecommendationClick = (recommendation: string) => {
    setInputText(recommendation);
  };

  return (
    <div className="min-h-screen flex flex-col justify-between p-8">
      <main className="flex-1 flex flex-col items-center gap-8">
        {error && (
          <div className="w-full max-w-2xl p-4 bg-red-100 text-red-700 rounded-lg">
            {error}
          </div>
        )}

        {generatedImage && (
          <Card className="w-full max-w-2xl">
            <CardContent className="p-4">
              <img
                src={generatedImage}
                alt="Generated artwork"
                className="w-full h-auto rounded-lg"
              />
            </CardContent>
          </Card>
        )}

        {recommendations.length > 0 && (
          <div className="w-full max-w-2xl">
            <h3 className="text-lg font-semibold mb-2">Try these prompts:</h3>
            <div className="flex flex-wrap gap-2">
              {recommendations.map((recommendation, index) => (
                <button
                  key={index}
                  onClick={() => handleRecommendationClick(recommendation)}
                  className="px-3 py-1 bg-black/[.05] dark:bg-white/[.06] rounded-full 
                           hover:bg-black/[.1] dark:hover:bg-white/[.1] transition-colors
                           text-sm"
                >
                  {recommendation}
                </button>
              ))}
            </div>
          </div>
        )}
      </main>

      <footer className="w-full max-w-3xl mx-auto">
        <form onSubmit={handleSubmit} className="w-full">
          <div className="flex gap-2">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="flex-1 p-3 rounded-lg bg-black/[.05] dark:bg-white/[.06] 
                       border border-black/[.08] dark:border-white/[.145] 
                       focus:outline-none focus:ring-2 focus:ring-black dark:focus:ring-white"
              placeholder="Describe the image you want to generate..."
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading}
              className="px-6 py-3 rounded-lg bg-foreground text-background 
                       hover:bg-[#383838] dark:hover:bg-[#ccc] transition-colors 
                       disabled:opacity-50"
            >
              {isLoading ? 'Generating...' : 'Generate'}
            </button>
          </div>
        </form>
      </footer>
    </div>
  );
}
