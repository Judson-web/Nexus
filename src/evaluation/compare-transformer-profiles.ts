export interface TransformerProfileReport {
  readonly dataset: { readonly sourceSha256: string; readonly tokenCount: number };
  readonly model: {
    readonly family: string;
    readonly config: Record<string, unknown>;
    readonly parameterCount: number;
    readonly epochs: number;
    readonly learningRate: number;
    readonly repeats: number;
  };
  readonly profile: {
    readonly totalDurationMs: number;
    readonly meanEpochDurationMs: number;
    readonly meanExamplesPerSecond: number;
    readonly medianFinalLoss: number;
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`Invalid profile report: ${name} must be a finite number.`);
  }
  return value;
}

export function parseTransformerProfileReport(value: unknown): TransformerProfileReport {
  if (!isRecord(value) || !isRecord(value.dataset) || !isRecord(value.model) || !isRecord(value.profile)) {
    throw new TypeError("Invalid profile report: expected dataset, model, and profile objects.");
  }
  const dataset = value.dataset;
  const model = value.model;
  const profile = value.profile;
  if (typeof dataset.sourceSha256 !== "string" || !/^[a-f0-9]{64}$/.test(dataset.sourceSha256)) {
    throw new TypeError("Invalid profile report: dataset.sourceSha256 must be a SHA-256 hex digest.");
  }
  if (!isRecord(model.config)) throw new TypeError("Invalid profile report: model.config must be an object.");
  if (typeof model.family !== "string") throw new TypeError("Invalid profile report: model.family must be a string.");
  return {
    dataset: {
      sourceSha256: dataset.sourceSha256,
      tokenCount: finiteNumber(dataset.tokenCount, "dataset.tokenCount"),
    },
    model: {
      family: model.family,
      config: model.config,
      parameterCount: finiteNumber(model.parameterCount, "model.parameterCount"),
      epochs: finiteNumber(model.epochs, "model.epochs"),
      learningRate: finiteNumber(model.learningRate, "model.learningRate"),
      repeats: finiteNumber(model.repeats, "model.repeats"),
    },
    profile: {
      totalDurationMs: finiteNumber(profile.totalDurationMs, "profile.totalDurationMs"),
      meanEpochDurationMs: finiteNumber(profile.meanEpochDurationMs, "profile.meanEpochDurationMs"),
      meanExamplesPerSecond: finiteNumber(profile.meanExamplesPerSecond, "profile.meanExamplesPerSecond"),
      medianFinalLoss: finiteNumber(profile.medianFinalLoss, "profile.medianFinalLoss"),
    },
  };
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (isRecord(value)) {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export function compareTransformerProfileReports(baselineInput: unknown, currentInput: unknown) {
  const baseline = parseTransformerProfileReport(baselineInput);
  const current = parseTransformerProfileReport(currentInput);
  const compatibilityFields: (keyof TransformerProfileReport["model"])[] = [
    "family", "parameterCount", "epochs", "learningRate", "repeats",
  ];
  for (const field of compatibilityFields) {
    if (stableJson(baseline.model[field]) !== stableJson(current.model[field])) {
      throw new Error(`Cannot compare profiles: model.${field} differs between reports.`);
    }
  }
  if (stableJson(baseline.model.config) !== stableJson(current.model.config)) {
    throw new Error("Cannot compare profiles: model.config differs between reports.");
  }
  if (baseline.dataset.sourceSha256 !== current.dataset.sourceSha256 ||
      baseline.dataset.tokenCount !== current.dataset.tokenCount) {
    throw new Error("Cannot compare profiles: dataset fingerprint or token count differs.");
  }

  const metric = (before: number, after: number) => ({
    baseline: before,
    current: after,
    absoluteChange: after - before,
    percentChange: before === 0 ? null : ((after - before) / Math.abs(before)) * 100,
  });
  return {
    compatible: true as const,
    datasetSha256: baseline.dataset.sourceSha256,
    tokenCount: baseline.dataset.tokenCount,
    model: { family: baseline.model.family, parameterCount: baseline.model.parameterCount, epochs: baseline.model.epochs, repeats: baseline.model.repeats },
    metrics: {
      totalDurationMs: metric(baseline.profile.totalDurationMs, current.profile.totalDurationMs),
      meanEpochDurationMs: metric(baseline.profile.meanEpochDurationMs, current.profile.meanEpochDurationMs),
      meanExamplesPerSecond: metric(baseline.profile.meanExamplesPerSecond, current.profile.meanExamplesPerSecond),
      medianFinalLoss: metric(baseline.profile.medianFinalLoss, current.profile.medianFinalLoss),
    },
    interpretation: "Negative duration changes mean faster timing; positive throughput changes mean higher throughput; negative final-loss changes mean lower training loss. Timing is environment-dependent and does not establish production performance.",
  };
}
