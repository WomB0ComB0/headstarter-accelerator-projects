import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { text } = body;

    if (!text || typeof text !== 'string') {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid prompt provided',
        },
        { status: 400 },
      );
    }

    const response = await fetch(`${process.env.BACKEND_URL}/api/generate-image`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to generate image');
    }

    const data = await response.json();

    if (data.image) {
      return NextResponse.json({
        success: true,
        image: data.image,
      });
    }

    throw new Error('No image generated');
  } catch (error) {
    console.error('Image generation error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to generate image',
      },
      { status: 500 },
    );
  }
}

export async function GET(request: Request) {
  try {
    const userId = request.headers.get('x-user-id');

    if (!userId) {
      return NextResponse.json(
        {
          success: false,
          error: 'User ID required',
        },
        { status: 400 },
      );
    }

    const response = await fetch(`${process.env.BACKEND_URL}/api/recommendations`, {
      headers: {
        'user-id': userId,
      },
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
      { status: 500 },
    );
  }
}
