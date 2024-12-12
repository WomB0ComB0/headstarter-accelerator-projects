import { Groq } from "groq-sdk";
import * as cheerio from 'cheerio';
import { z } from 'zod'
import { ChatCompletionMessageParam } from "groq-sdk/resources/chat/completions.mjs";

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY || (() =>  {throw new Error('GROQ_API_KEY is not set')})()
});

function extractSemanticContent($: cheerio.CheerioAPI): string {
  const semanticSelectors = [
    'article',
    'section',
    'main',
    'header',
    'footer',
    'nav',
    'aside',
    'details',
    'summary',
    'blockquote',
    'p',
    'div',
    'span',
    'a',
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'li',
    'meta[name="description"]',
    'meta[property="og:description"]',
    'title'
  ];

  let extractedText = '';
  
  // First extract meta information
  const title = $('title').text();
  const description = $('meta[name="description"]').attr('content');
  if (title) extractedText += `Title: ${title}\n\n`;
  if (description) extractedText += `Description: ${description}\n\n`;

  // Then extract content from semantic elements
  semanticSelectors.forEach(selector => {
    const elements = $(selector);
    
    elements.each((_, element) => {
      let rawText = '';
      if (element.type === 'tag' && element.name === 'meta') {
        rawText = $(element).attr('content') || '';
      } else {
        rawText = $(element).text();
      }
      
      const cleanedText = rawText
        .trim()
        .replace(/[\u{1F600}-\u{1F64F}]/gu, '')
        .replace(/[^\x00-\x7F]/g, '')
        .replace(/\s+/g, ' ')
        .replace(/[\n\r]+/g, ' ');

      if (cleanedText.length > 50) {
        extractedText += cleanedText + '\n\n';
      }
    });
  });

  return extractedText.trim() || $('body').text().trim();
}

const domainSpecificExtractors: Record<string, (($: cheerio.CheerioAPI) => string)> = {
  'wikipedia.org': ($) => $('#mw-content-text').text().trim(),
  'github.com': ($) => $('.markdown-body').text().trim(),
  'medium.com': ($) => $('article > div > div > section').text().trim()
};

let urlString: string

export async function POST(req: Request) {
  try {
    const schema = z.object({
      messages: z.array(z.object({ role: z.string(), content: z.string() })),
      url: z.string().url()
    });
    const { messages, url } = await schema.parse(await req.json());
    urlString = url
    
    if (!urlString || !urlString.startsWith('http')) {
      throw new Error('Invalid URL');
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(urlString, { 
      signal: controller.signal,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache'
      }
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const html = await response.text();
    const $ = cheerio.load(html);

    const hostname = new URL(urlString).hostname;
    const specificExtractor = Object.entries(domainSpecificExtractors)
      .find(([domain]) => hostname.includes(domain))?.[1];

    let pageContent = specificExtractor 
      ? specificExtractor($) 
      : extractSemanticContent($);

    console.log(pageContent);

    const MAX_CONTENT_LENGTH = 8000;
    pageContent = pageContent.slice(0, MAX_CONTENT_LENGTH);

    const systemMessage = `You are a helpful AI assistant. Use the following webpage content to answer questions precisely and concisely:

${pageContent}

Answer questions strictly based on this content. If the answer is not in the text, clearly state that.`;

    const completion = await groq.chat.completions.create({
      messages: [
        { role: "system", content: systemMessage },
        ...messages
      ] as ChatCompletionMessageParam[],
      model: "mixtral-8x7b-32768",
      temperature: 0.5,
      max_tokens: 2048,
    });

    return Response.json({
      message: completion.choices[0]?.message?.content || "No response generated",
      sourceUrl: urlString
    });

  } catch (error) {
    console.error('Web Scraping & Chat API Error:', error);
    return Response.json(
      { 
        error: error instanceof Error ? error.message : "Failed to process request",
        sourceUrl: urlString 
      },
      { status: 500 }
    );
  }
}