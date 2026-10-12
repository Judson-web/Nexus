"use client";

import { FormEvent, useState } from "react";

type Result = { text?: string; model?: string; note?: string; error?: string };
const examples = [
  { label: "How models learn", prompt: "A language model learns by" },
  { label: "Independent AI", prompt: "The future of independent AI" },
  { label: "Better experiments", prompt: "Reliable experiments require" },
];

export default function HomePage() {
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(80);
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
        <h1>Intelligence,<br /><span>built from the ground up.</span></h1>
        <p className="intro">We’re building an original language model one verified step at a time. Our code, our training process, our inference runtime — with no third-party AI inference APIs.</p>
        <div className="hero-meta" aria-label="Project principles">
          <span><i /> SELF-HOSTED INFERENCE</span>
          <span><i /> TYPESCRIPT FIRST</span>
          <span><i /> RESEARCH IN PROGRESS</span>
        </div>
      </section>

      <section className="workspace" aria-labelledby="playground-title">
        <div className="workspace-heading">
          <div><span className="kicker">01 / PLAYGROUND</span><h2 id="playground-title">Give it a starting point.</h2></div>
          <span className="runtime-chip"><i /> LOCAL MODEL</span>
        </div>
        <p className="section-copy">Try a short prompt and inspect what the current baseline predicts next. This is an early experiment, not a general-purpose assistant.</p>
        <form onSubmit={generate}>
          <div className="field-heading">
            <label htmlFor="prompt">YOUR PROMPT</label>
            <span className="field-hint">Up to 160 characters</span>
          </div>
          <textarea id="prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Start with a phrase, idea, or sentence…" maxLength={160} rows={4} aria-describedby="prompt-meta" />
          <div className="prompt-meta" id="prompt-meta"><span>Shorter prompts are easier to experiment with.</span><span>{prompt.length}/160</span></div>
          <div className="suggestion-row">
            <span className="suggestion-label">TRY A STARTER</span>
            <div className="suggestions">{examples.map((example) => <button className="suggestion" type="button" key={example.prompt} onClick={() => setPrompt(example.prompt)}>{example.label}</button>)}</div>
          </div>
          <div className="form-actions">
            <label className="token-control" htmlFor="tokens"><span>OUTPUT LENGTH</span><select id="tokens" value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))}><option value={40}>Short · 40</option><option value={80}>Medium · 80</option><option value={120}>Long · 120</option><option value={160}>Max · 160</option></select></label>
            <button className="generate" type="submit" disabled={!prompt.trim() || busy}>{busy ? <><span className="spinner" /> Generating…</> : <>Run inference <span aria-hidden="true">↗</span></>}</button>
          </div>
        </form>
        {(busy || result.text || result.error) && <section className="output" aria-live="polite" aria-busy={busy}>
          <div className="output-heading"><span>MODEL OUTPUT</span>{result.model && <span className="model-tag">{result.model}</span>}</div>
          {busy && <div className="loading-state"><span className="spinner" /><p>Running the current model against your prompt…</p></div>}
          {result.error && <p className="error" role="alert">{result.error}</p>}
          {result.text && <><p className="generated">{result.text}</p><div className="output-footer"><span>EXPERIMENTAL OUTPUT</span><button type="button" className="text-button" onClick={() => void navigator.clipboard?.writeText(result.text ?? "")}>Copy output ↗</button></div></>}
          {result.note && <p className="note">{result.note}</p>}
        </section>}
      </section>

      <section className="approach" id="approach">
        <div className="approach-heading"><span className="kicker">02 / THE APPROACH</span><h2>Capability is earned, not claimed.</h2><p>We’re focused on the foundations that make an independently trained model worth using.</p></div>
        <div className="principles">
          <article><span>01</span><div><h3>Own the stack</h3><p>Inference runs in our own code. No external model API hiding behind the interface.</p></div></article>
          <article><span>02</span><div><h3>Train with intent</h3><p>Curated datasets, reproducible configurations, and held-out evaluation.</p></div></article>
          <article><span>03</span><div><h3>Show the limits</h3><p>Today’s baseline is deliberately small. Better data and training must come before bigger claims.</p></div></article>
        </div>
      </section>

      <footer><a className="footer-brand" href="/">NEXUS<span className="accent">.</span></a><span>INDEPENDENT AI RESEARCH</span><span>EARLY BUILD · NOT A GENERAL-PURPOSE ASSISTANT</span></footer>
    </main>
  );
}
