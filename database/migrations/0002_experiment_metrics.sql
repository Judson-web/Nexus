BEGIN;

ALTER TABLE public.datasets
  ADD COLUMN IF NOT EXISTS split_manifest jsonb NOT NULL DEFAULT '{}'::jsonb
  CHECK (jsonb_typeof(split_manifest) = 'object');

ALTER TABLE public.training_runs
  ADD COLUMN IF NOT EXISTS initial_loss double precision
    CHECK (initial_loss IS NULL OR initial_loss >= 0),
  ADD COLUMN IF NOT EXISTS validation_loss double precision
    CHECK (validation_loss IS NULL OR validation_loss >= 0),
  ADD COLUMN IF NOT EXISTS test_loss double precision
    CHECK (test_loss IS NULL OR test_loss >= 0),
  ADD COLUMN IF NOT EXISTS metrics jsonb NOT NULL DEFAULT '{}'::jsonb
    CHECK (jsonb_typeof(metrics) = 'object');

INSERT INTO public.schema_migrations (version)
VALUES ('0002_experiment_metrics')
ON CONFLICT (version) DO NOTHING;

COMMIT;
