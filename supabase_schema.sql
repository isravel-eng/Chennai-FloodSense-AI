-- =============================================================================
-- Chennai FloodSense AI — Supabase PostgreSQL Schema
-- Run this entire script once in the Supabase SQL Editor.
-- It is idempotent (uses IF NOT EXISTS / CREATE OR REPLACE).
-- =============================================================================


-- ---------------------------------------------------------------------------
-- 1. USER PROFILES
--    auth.users is managed by Supabase Auth.
--    public.users stores the application-level profile linked by UUID.
-- ---------------------------------------------------------------------------
create table if not exists public.users (
  id              uuid        primary key references auth.users(id) on delete cascade,
  name            text        not null,
  email           text        unique not null,
  native_locality text        not null,
  created_at      timestamptz default now()
);

-- Row Level Security: each user may only read/write their own row.
alter table public.users enable row level security;

-- Drop policies before recreating (idempotent)
drop policy if exists "users_self_read"   on public.users;
drop policy if exists "users_self_insert" on public.users;
drop policy if exists "users_self_update" on public.users;

create policy "users_self_read" on public.users
  for select using (auth.uid() = id);

create policy "users_self_insert" on public.users
  for insert with check (auth.uid() = id);

create policy "users_self_update" on public.users
  for update using (auth.uid() = id);


-- ---------------------------------------------------------------------------
-- 2. RAINFALL OBSERVATIONS
--    Replaces data/processed/live_rainfall_log.csv.
--
--    Granularity: DAILY.
--    Rationale: the ML pipeline uses daily rainfall_mm values and computes
--    rolling sums / lag features over whole days.  Storing sub-daily
--    observations would require aggregation that the existing code does not
--    perform and would silently change model input semantics.
--
--    Uniqueness: one row per (locality, observed_at).
--    UPSERT on conflict keeps the most-recent value.
-- ---------------------------------------------------------------------------
create table if not exists public.rainfall_logs (
  id           bigint      generated always as identity primary key,
  locality     text        not null,
  observed_at  date        not null,
  rainfall_mm  double precision not null,
  source       text        not null default 'open_meteo',
  -- source values: 'open_meteo' | 'manual' | 'imported'
  created_at   timestamptz default now(),
  constraint rainfall_logs_locality_date_unique unique (locality, observed_at)
);

-- Performance index for the time-series queries used by the ML pipeline.
create index if not exists idx_rainfall_logs_locality_date
  on public.rainfall_logs (locality, observed_at desc);

-- Row Level Security
alter table public.rainfall_logs enable row level security;

drop policy if exists "rainfall_read_authenticated" on public.rainfall_logs;
drop policy if exists "rainfall_write_service_role"  on public.rainfall_logs;

-- Any authenticated user (or service role) can read rainfall logs.
-- This allows the frontend to show historical data for any locality.
create policy "rainfall_read_authenticated" on public.rainfall_logs
  for select using (
    auth.role() in ('authenticated', 'service_role')
  );

-- Only the backend service role may write (prevents client-side injection).
create policy "rainfall_write_service_role" on public.rainfall_logs
  for insert with check (auth.role() = 'service_role');

create policy "rainfall_upsert_service_role" on public.rainfall_logs
  for update using (auth.role() = 'service_role');


-- ---------------------------------------------------------------------------
-- 3. MIGRATION / IMPORT GUIDANCE
--
--    If existing live_rainfall_log.csv data must be preserved, run this
--    pattern from psql or the Supabase SQL editor after uploading the CSV:
--
--    COPY public.rainfall_logs (locality, observed_at, rainfall_mm, source)
--    FROM '/path/to/live_rainfall_log.csv'
--    CSV HEADER;
--
--    Alternatively, use the Python migration script described in README.md.
-- ---------------------------------------------------------------------------
