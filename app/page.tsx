"use client";

import { FormEvent, useState } from "react";

type Result = { text?: string; model?: string; note?: string; error?: string };
const examples = ["A language model learns by", "The future of independent AI", "Reliable experiments require"];

export default function HomePage() {
  const [prompt, setPrompt] = useState("");
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
        body: JSON.stringify({ prompt: prompt.trim(), maxTokens: 100 }),
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

  return <main className="shell">
    <header className="topbar">
      <a className="brand" href="/" aria-label="Nexus home"><span className="brand-mark">N</span><span>NEXUS<span className="accent">.</span></span></a>
      <span className="status"><i /> OWN INFERENCE RUNTIME</span>
    </header>
    <section className="hero">
      <div className="eyebrow"><span className="eyebrow-line" /> INDEPENDENT MODEL LAB</div>
      <h1>Intelligence,<br /><span>built from the ground up.</span></h1>
      <p className="intro">A small first step toward an independently trained language model. No third-party inference APIs. Just our code, our data, and measurable progress.</p>
    </section>
    <section className="workspace" aria-labelledby="playground-title">
      <div className="workspace-heading"><div><span className="kicker">01 / PLAYGROUND</span><h2 id="playground-title">Try the model</h2></div><span className="runtime-chip"><i /> SELF-HOSTED</span></div>
      <form onSubmit={generate}>
        <label htmlFor="prompt">YOUR PROMPT</label>
        <textarea id="prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Give the model a starting point..." maxLength={160} rows={4} />
        <div className="prompt-meta"><span>Keep it short for this early experiment.</span><span>{prompt.length}/160</span></div>
        <div className="suggestions" aria-label="Prompt suggestions">{examples.map((example) => <button className="suggestion" type="button" key={example} onClick={() => setPrompt(example)}>{example}</button>)}</div>
        <button className="generate" type="submit" disabled={!prompt.trim() || busy}>{busy ? <><span className="spinner" /> Running inference…</> : <>Generate <span aria-hidden="true">↗</span></>}</button>
      </form>
      {(busy || result.text || result.error) && <div className="output" aria-live="polite">
        <div className="output-heading"><span>MODEL OUTPUT</span>{result.model && <span className="model-tag">{result.model}</span>}</div>
        {busy && <p className="muted">Preparing the model and generating text…</p>}
        {result.error && <p className="error">{result.error}</p>}
        {result.text && <p className="generated">{result.text}</p>}
        {result.note && <p className="note">{result.note}</p>}
      </div>}
    </section>
    <section className="principles">
      <article><span>01</span><h3>Own the stack</h3><p>Model execution stays in our runtime. No external AI inference provider.</p></article>
      <article><span>02</span><h3>Train with intent</h3><p>Curated data, recorded configurations, reproducible experiments.</p></article>
      <article><span>03</span><h3>Earn capability</h3><p>Small, testable milestones before scaling architecture or claims.</p></article>
    </section>
    <footer><span>© {new Date().getFullYear()} NEXUS LAB</span><span>EARLY RESEARCH BUILD · NOT A GENERAL-PURPOSE ASSISTANT</span></footer>
  </main>;
}
