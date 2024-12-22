import { NextResponse } from 'next/server';
import crypto from 'crypto';

const MODAL_API = {
  GENERATE: 'https://womb0comb0--image-generation-model-generate.modal.run',
  RECOMMENDATIONS: 'https://womb0comb0--image-generation-model-get-recommendations.modal.run',
  HEALTH: 'https://womb0comb0--image-generation-model-health.modal.run'
};

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { text } = body;

    if (!text || typeof text !== 'string') {
      return NextResponse.json(
        { success: false, error: 'Invalid prompt provided' },
        { status: 400 }
      );
    }

    const imageId = crypto.randomUUID();

    const response = await fetch(MODAL_API.GENERATE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        prompt: text,
        image_id: imageId 
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to generate image');
    }

    const data = await response.json();

    if (!data.image) {
      throw new Error(data.error || 'No image generated');
    }

    return NextResponse.json({
      success: true,
      image: data.image,
      imageId: imageId,
      cached: data.cached,
      recommendations: data.recommendations,
      safetyCheck: data.safety_check
    });
  } catch (error) {
    console.error('Image generation error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to generate image',
      },
      { status: 500 }
    );
  }
}

export async function GET(request: Request) {
  try {
    const body = await request.json();
    const { imageId } = body;

    const response = await fetch(MODAL_API.RECOMMENDATIONS, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_id: imageId })
    });

    if (!response.ok) {
      throw new Error('Failed to fetch recommendations');
    }

    const data = await response.json();
    return NextResponse.json({
      success: true,
      recommendations: data.recommendations,
    });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch recommendations',
      },
      { status: 500 }
    );
  }
}
