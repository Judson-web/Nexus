import { Pool } from "pg";

export interface TrainingRunRecord {
  readonly modelFamily: string;
  readonly config: Record<string, unknown>;
  readonly seed: number;
}

let pool: Pool | undefined;

function getPool(): Pool | undefined {
  const connectionString = process.env.DATABASE_URL;
  if (!connectionString) return undefined;
  pool ??= new Pool({
    connectionString,
    max: 2,
    connectionTimeoutMillis: 5000,
    idleTimeoutMillis: 10000,
    ssl: connectionString.includes("localhost") ? undefined : { rejectUnauthorized: true },
  });
  return pool;
}

/** Optional metadata tracking. Training remains usable without DATABASE_URL. */
export async function startTrainingRun(record: TrainingRunRecord): Promise<string | undefined> {
  const clientPool = getPool();
  if (!clientPool) return undefined;
  const result = await clientPool.query<{ id: string }>(
    `INSERT INTO public.training_runs
      (model_family, config, seed, status, started_at)
     VALUES ($1, $2::jsonb, $3, 'running', now())
     RETURNING id`,
    [record.modelFamily, JSON.stringify(record.config), record.seed],
  );
  return result.rows[0]?.id;
}

export async function finishTrainingRun(
  id: string | undefined,
  result: { finalLoss: number },
): Promise<void> {
  if (!id) return;
  const clientPool = getPool();
  if (!clientPool) return;
  await clientPool.query(
    `UPDATE public.training_runs
        SET status = 'succeeded', final_loss = $2, finished_at = now()
      WHERE id = $1 AND status = 'running'`,
    [id, result.finalLoss],
  );
}

export async function failTrainingRun(id: string | undefined, error: unknown): Promise<void> {
  if (!id) return;
  const clientPool = getPool();
  if (!clientPool) return;
  const summary = (error instanceof Error ? error.message : String(error)).slice(0, 2000);
  await clientPool.query(
    `UPDATE public.training_runs
        SET status = 'failed', error_summary = $2, finished_at = now()
      WHERE id = $1 AND status = 'running'`,
    [id, summary],
  );
}
