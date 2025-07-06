/*
  # Analysis Cache Table

  1. New Tables
    - `analysis_cache`
      - `id` (uuid, primary key)
      - `cache_key` (text, unique)
      - `result` (jsonb)
      - `created_at` (timestamp)
      - `expires_at` (timestamp)

  2. Security
    - Enable RLS on `analysis_cache` table
    - Add policy for service access
*/

CREATE TABLE IF NOT EXISTS analysis_cache (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  cache_key text UNIQUE NOT NULL,
  result jsonb NOT NULL,
  created_at timestamptz DEFAULT now(),
  expires_at timestamptz DEFAULT (now() + interval '1 hour')
);

ALTER TABLE analysis_cache ENABLE ROW LEVEL SECURITY;

-- Create index for efficient cache lookups
CREATE INDEX IF NOT EXISTS idx_analysis_cache_key ON analysis_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_analysis_cache_expires ON analysis_cache(expires_at);

-- Policy for service access (adjust based on your auth setup)
CREATE POLICY "Service can manage cache"
  ON analysis_cache
  FOR ALL
  TO service_role
  USING (true);

-- Function to clean expired cache entries
CREATE OR REPLACE FUNCTION clean_expired_cache()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
  DELETE FROM analysis_cache WHERE expires_at < now();
END;
$$;