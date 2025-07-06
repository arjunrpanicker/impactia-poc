-- Enable the pgvector extension to work with embeddings
create extension if not exists vector;

-- Create an enum for code types
create type code_type as enum ('file', 'method', 'class');

-- Create the code_embeddings table
create table code_embeddings (
    id bigint primary key generated always as identity,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    embedding vector(1536), -- OpenAI embeddings are 1536 dimensions
    code_type code_type not null,
    metadata jsonb not null default '{}'::jsonb,
    content text not null,
    file_path text not null,
    repository text not null,
    branch text not null default 'main',
    last_modified timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create an index for similarity search
create index on code_embeddings 
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

-- Function to match similar code embeddings
create or replace function match_code_embeddings (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    code_embeddings.id,
    code_embeddings.content,
    code_embeddings.metadata,
    1 - (code_embeddings.embedding <=> query_embedding) as similarity
  from code_embeddings
  where 1 - (code_embeddings.embedding <=> query_embedding) > match_threshold
  order by code_embeddings.embedding <=> query_embedding
  limit match_count;
end;
$$; 