declare module 'react-intersection-observer' {
  export function useInView(options?: {
    threshold?: number | number[];
    root?: Element | null;
    rootMargin?: string;
    triggerOnce?: boolean;
  }): {
    ref: (node?: Element | null) => void;
    inView: boolean;
    entry?: IntersectionObserverEntry;
  };
}

declare module 'lucide-react' {
  import { FC, SVGProps } from 'react';
  export const Heart: FC<SVGProps<SVGSVGElement>>;
  export const MessageCircle: FC<SVGProps<SVGSVGElement>>;
  export const Share2: FC<SVGProps<SVGSVGElement>>;
  export const Home: FC<SVGProps<SVGSVGElement>>;
  export const Compass: FC<SVGProps<SVGSVGElement>>;
  export const Upload: FC<SVGProps<SVGSVGElement>>;
  export const User: FC<SVGProps<SVGSVGElement>>;
} 