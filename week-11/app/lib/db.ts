import { createClient } from '@supabase/supabase-js';
import { env } from '@/config/env';
import { Database } from '@/types/supabase';

export const createServerClient = () => 
  createClient<Database>(
    env.NEXT_PUBLIC_SUPABASE_URL,
    env.SUPABASE_SERVICE_ROLE_KEY
  );

export const createClientClient = () =>
  createClient<Database>(
    env.NEXT_PUBLIC_SUPABASE_URL,
    env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  ); 