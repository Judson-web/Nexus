BEGIN;

CREATE TABLE IF NOT EXISTS public.schema_migrations (
  version text PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.datasets (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL CHECK (length(name) BETWEEN 1 AND 160),
  version text NOT NULL,
  source_description text NOT NULL,
  license text NOT NULL,
  content_sha256 text NOT NULL CHECK (content_sha256 ~ '^[a-f0-9]{64}$'),
  example_count bigint NOT NULL DEFAULT 0 CHECK (example_count >= 0),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS public.dataset_examples (
  id bigserial PRIMARY KEY,
  dataset_id uuid NOT NULL REFERENCES public.datasets(id) ON DELETE CASCADE,
  example_sha256 text NOT NULL CHECK (example_sha256 ~ '^[a-f0-9]{64}$'),
  split text NOT NULL CHECK (split IN ('train', 'validation', 'test')),
  token_count integer NOT NULL CHECK (token_count >= 0),
  content_uri text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (dataset_id, example_sha256)
);

CREATE INDEX IF NOT EXISTS dataset_examples_split_idx
  ON public.dataset_examples (dataset_id, split);

CREATE TABLE IF NOT EXISTS public.training_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_id uuid REFERENCES public.datasets(id) ON DELETE SET NULL,
  model_family text NOT NULL,
  config jsonb NOT NULL CHECK (jsonb_typeof(config) = 'object'),
  seed bigint NOT NULL,
  status text NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
  started_at timestamptz,
  finished_at timestamptz,
  final_loss double precision CHECK (final_loss IS NULL OR final_loss >= 0),
  error_summary text,
  created_at timestamptz NOT NULL DEFAULT now(),
  CHECK (finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at)
);

CREATE INDEX IF NOT EXISTS training_runs_created_idx ON public.training_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS training_runs_status_idx ON public.training_runs (status, created_at DESC);

CREATE TABLE IF NOT EXISTS public.model_versions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  training_run_id uuid REFERENCES public.training_runs(id) ON DELETE SET NULL,
  model_family text NOT NULL,
  version text NOT NULL,
  checkpoint_uri text NOT NULL,
  checkpoint_sha256 text NOT NULL CHECK (checkpoint_sha256 ~ '^[a-f0-9]{64}$'),
  config jsonb NOT NULL CHECK (jsonb_typeof(config) = 'object'),
  parameter_count bigint NOT NULL CHECK (parameter_count >= 0),
  status text NOT NULL CHECK (status IN ('candidate', 'evaluated', 'approved', 'retired')),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (model_family, version)
);

CREATE TABLE IF NOT EXISTS public.evaluation_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_version_id uuid NOT NULL REFERENCES public.model_versions(id) ON DELETE CASCADE,
  dataset_id uuid REFERENCES public.datasets(id) ON DELETE SET NULL,
  metrics jsonb NOT NULL CHECK (jsonb_typeof(metrics) = 'object'),
  passed boolean NOT NULL,
  notes text NOT NULL DEFAULT '',
  evaluated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO public.schema_migrations (version)
VALUES ('0001_foundation')
ON CONFLICT (version) DO NOTHING;

COMMIT;
