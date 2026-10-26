"use client";

import { FormEvent, useState } from "react";

type Result = { text?: string; model?: string; provider?: string; note?: string; error?: string };
const examples = [
  { label: "Explain a concept", prompt: "Explain photosynthesis in simple terms, including what plants take in and what they produce." },
  { label: "Think it through", prompt: "What are three practical ways to learn a difficult subject more effectively? Explain why each helps." },
  { label: "Build something", prompt: "Help me design a small TypeScript project that teaches me how language models generate text." },
];

export default function HomePage() {
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(160);
  const [result, setResult] = useState<Result>({});
  const [busy, setBusy] = useState(false);

  async function generate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!prompt.trim() || busy) return;
    setBusy(true);
    setResult({});
    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt.trim(), maxTokens }),
      });
      const data = (await response.json()) as Result;
      if (!response.ok) throw new Error(data.error ?? "Generation failed. Try again.");
      setResult(data);
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : "Something went wrong." });
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Nexus home">
          <span className="brand-mark" aria-hidden="true">N</span>
          <span>NEXUS<span className="accent">.</span></span>
        </a>
        <a className="top-link" href="#approach">OUR APPROACH <span aria-hidden="true">↘</span></a>
      </header>

      <section className="hero">
        <div className="eyebrow"><span className="eyebrow-line" /> INDEPENDENT MODEL LAB</div>
        <h1>Intelligence,<br /><span>built with evidence.</span></h1>
        <p className="intro">Nexus is building an original training and evaluation stack. While our from-scratch model is still experimental, the playground can serve a real pretrained open-weight instruction model through a configured inference provider. The runtime tells you which model answered.</p>
        <div className="hero-meta" aria-label="Project principles">
          <span><i /> OPEN-WEIGHT MODELS</span>
          <span><i /> REPRODUCIBLE TRAINING</span>
          <span><i /> RESEARCH IN PROGRESS</span>
        </div>
      </section>

      <section className="workspace" aria-labelledby="playground-title">
        <div className="workspace-heading">
          <div><span className="kicker">01 / PLAYGROUND</span><h2 id="playground-title">Ask Nexus a question.</h2></div>
          <span className="runtime-chip"><i /> OPEN-MODEL PLAYGROUND</span>
        </div>
        <p className="section-copy">Ask a complete question or give a clear instruction. Connect Hugging Face Inference Providers with a server-side HF_TOKEN, or connect a private OpenAI-compatible endpoint. See the setup guide if the provider is not configured.</p>
        <form onSubmit={generate}>
          <div className="field-heading">
            <label htmlFor="prompt">YOUR QUESTION OR INSTRUCTION</label>
            <span className="field-hint">Up to 4,000 characters</span>
          </div>
          <textarea id="prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="What would you like to understand or create?" maxLength={4000} rows={5} aria-describedby="prompt-meta" />
          <div className="prompt-meta" id="prompt-meta"><span>Clear questions give the model a better target.</span><span>{prompt.length}/4000</span></div>
          <div className="suggestion-row">
            <span className="suggestion-label">TRY A STARTER</span>
            <div className="suggestions">{examples.map((example) => <button className="suggestion" type="button" key={example.prompt} onClick={() => setPrompt(example.prompt)}>{example.label}</button>)}</div>
          </div>
          <div className="form-actions">
            <label className="token-control" htmlFor="tokens"><span>OUTPUT LENGTH</span><select id="tokens" value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))}><option value={80}>Short · 80</option><option value={160}>Medium · 160</option><option value={256}>Long · 256</option><option value={512}>Max · 512</option></select></label>
            <button className="generate" type="submit" disabled={!prompt.trim() || busy}>{busy ? <><span className="spinner" /> Generating…</> : <>Ask Nexus <span aria-hidden="true">↗</span></>}</button>
          </div>
        </form>
        {(busy || result.text || result.error) && <section className="output" aria-live="polite" aria-busy={busy}>
          <div className="output-heading"><span>MODEL RESPONSE</span>{result.model && <span className="model-tag">{result.model}</span>}</div>
          {busy && <div className="loading-state"><span className="spinner" /><p>Waiting for the configured language model…</p></div>}
          {result.error && <p className="error" role="alert">{result.error}</p>}
          {result.text && <><p className="generated">{result.text}</p><div className="output-footer"><span>AI-GENERATED · VERIFY IMPORTANT CLAIMS</span><button type="button" className="text-button" onClick={() => void navigator.clipboard?.writeText(result.text ?? "")}>Copy response ↗</button></div></>}
          {result.provider && <p className="note">Provider: {result.provider}</p>}
          {result.note && <p className="note">{result.note}</p>}
        </section>}
        <p className="note">Privacy note: when using a hosted provider, your prompt is sent to that provider for inference. For private serving, configure an endpoint you control.</p>
      </section>

      <section className="approach" id="approach">
        <div className="approach-heading"><span className="kicker">02 / THE APPROACH</span><h2>Capability is earned, not claimed.</h2><p>Useful inference and original model research are separate tracks. One does not pretend to be the other.</p></div>
        <div className="principles">
          <article><span>01</span><div><h3>Choose the runtime</h3><p>Use Hugging Face Inference Providers or connect a self-hosted OpenAI-compatible server such as llama.cpp, vLLM, or SGLang.</p></div></article>
          <article><span>02</span><div><h3>Train with intent</h3><p>Keep data provenance, license checks, split hygiene, reproducible settings, and held-out evaluation in the training path.</p></div></article>
          <article><span>03</span><div><h3>Label the limits</h3><p>The tiny from-scratch Transformer remains a research artifact. It is not promoted as a capable assistant without evidence.</p></div></article>
        </div>
      </section>

      <footer><a className="footer-brand" href="/">NEXUS<span className="accent">.</span></a><span>OPEN-WEIGHT INFERENCE · ORIGINAL MODEL RESEARCH</span><span>EARLY BUILD · VERIFY IMPORTANT OUTPUTS</span></footer>
    </main>
  );
}
