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

/** Metadata persistence is opt-in; local training never requires Neon. */
export async function startTrainingRun(record: TrainingRunRecord): Promise<string | undefined> {
  const clientPool = getPool();
  if (!clientPool) return undefined;
  const result = await clientPool.query<{ id: string }>(
    `INSERT INTO public.training_runs (model_family, config, seed, status, started_at)
     VALUES ($1, $2::jsonb, $3, 'running', now()) RETURNING id`,
    [record.modelFamily, JSON.stringify(record.config), record.seed],
  );
  return result.rows[0]?.id;
}

export async function finishTrainingRun(
  id: string | undefined,
  result: { finalLoss: number; initialLoss?: number },
): Promise<void> {
  if (!id) return;
  const clientPool = getPool();
  if (!clientPool) return;
  await clientPool.query(
    `UPDATE public.training_runs
       SET status = 'succeeded', final_loss = $2, initial_loss = $3, finished_at = now()
     WHERE id = $1 AND status = 'running'`,
    [id, result.finalLoss, result.initialLoss ?? null],
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

export interface EvaluationRecord {
  readonly modelFamily: string;
  readonly datasetId?: string;
  readonly metrics: Record<string, unknown>;
  readonly notes?: string;
}

export async function recordEvaluation(records: readonly EvaluationRecord[]): Promise<void> {
  if (records.length === 0) return;
  const clientPool = getPool();
  if (!clientPool) return;
  const client = await clientPool.connect();
  try {
    await client.query("BEGIN");
    for (const record of records) {
      await client.query(
        `INSERT INTO public.evaluation_runs (model_version_id, dataset_id, metrics, passed, notes)
         SELECT mv.id, $2::uuid, $3::jsonb, false, $4
         FROM public.model_versions mv
         WHERE mv.model_family = $1
         ORDER BY mv.created_at DESC
         LIMIT 1`,
        [record.modelFamily, record.datasetId ?? null, JSON.stringify(record.metrics), record.notes ?? "Held-out diagnostic evaluation; not a production approval."],
      );
    }
    await client.query("COMMIT");
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}
