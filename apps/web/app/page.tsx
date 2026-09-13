"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Health = {
  status: string;
  device: string;
  dtype: string;
  ready: boolean;
};

type DetectResult = {
  is_synthetic: boolean;
  confidence: number;
};

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DetectResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const checkHealth = useCallback(async () => {
    setHealthError(null);
    try {
      const res = await fetch(`${API_URL}/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setHealth(await res.json());
    } catch (e: unknown) {
      setHealth(null);
      setHealthError(e instanceof Error ? e.message : "Connection failed");
    }
  }, []);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealthError("Connection failed"));
  }, []);

  const handleFile = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setFileName(file.name);

    try {
      const buf = await file.arrayBuffer();
      const bytes = new Uint8Array(buf);
      let binary = "";
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]!);
      }
      const audio_base64 = btoa(binary);

      const res = await fetch(`${API_URL}/detect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio_base64 }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail ?? `HTTP ${res.status}`);
      }
      setResult(data as DetectResult);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const onDragLeave = useCallback(() => setDragging(false), []);

  return (
    <div className="flex flex-col min-h-screen bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800 px-6 py-4">
        <h1 className="text-lg font-semibold tracking-tight">
          altur-vconstruct
        </h1>
        <p className="text-sm text-zinc-400">Voice anti-spoofing tester</p>
      </header>

      <main className="flex-1 flex flex-col items-center gap-8 px-4 py-10 max-w-2xl mx-auto w-full">
        {/* Health */}
        <section className="w-full rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-medium text-zinc-300 uppercase tracking-wider">
              API Health
            </h2>
            <button
              onClick={checkHealth}
              className="text-xs px-3 py-1 rounded-md bg-zinc-800 hover:bg-zinc-700 transition-colors"
            >
              Check
            </button>
          </div>
          {health && (
            <div className="flex items-center gap-4 text-sm">
              <span className="flex items-center gap-1.5">
                <span
                  className={`inline-block w-2 h-2 rounded-full ${health.ready ? "bg-emerald-400" : "bg-amber-400"}`}
                />
                {health.status}
              </span>
              <span className="text-zinc-500">|</span>
              <span className="text-zinc-400">device: {health.device}</span>
              <span className="text-zinc-500">|</span>
              <span className="text-zinc-400">dtype: {health.dtype}</span>
            </div>
          )}
          {healthError && (
            <p className="text-sm text-red-400">✗ {healthError}</p>
          )}
          {!health && !healthError && (
            <p className="text-sm text-zinc-500">Press Check to ping the API</p>
          )}
        </section>

        {/* Upload */}
        <section
          className={`w-full rounded-xl border-2 border-dashed p-10 text-center transition-colors cursor-pointer
            ${dragging ? "border-blue-500 bg-blue-500/10" : "border-zinc-700 hover:border-zinc-500 bg-zinc-900/30"}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".wav,audio/wav"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
              e.target.value = "";
            }}
          />
          <div className="flex flex-col items-center gap-2">
            <svg
              className="w-8 h-8 text-zinc-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z"
              />
            </svg>
            <p className="text-sm text-zinc-300">
              Drop stereo WAV here or <span className="underline">browse</span>
            </p>
            <p className="text-xs text-zinc-500">
              8 kHz, 16-bit PCM — ch0 caller, ch1 agent
            </p>
          </div>
        </section>

        {/* Loading */}
        {loading && (
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
                fill="none"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            Analyzing {fileName}…
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="w-full rounded-xl border border-red-900/50 bg-red-950/30 p-4 text-sm text-red-400">
            ✗ {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <section className="w-full rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
            <h2 className="text-sm font-medium text-zinc-300 uppercase tracking-wider mb-4">
              Result
            </h2>
            <div className="flex items-center gap-6">
              <div>
                <span
                  className="text-3xl font-bold"
                  style={{ color: result.is_synthetic ? "#f87171" : "#34d399" }}
                >
                  {result.is_synthetic ? "SYNTHETIC" : "HUMAN"}
                </span>
                <p className="text-xs text-zinc-500 mt-0.5">{fileName}</p>
              </div>
              <div className="text-right">
                <span className="text-2xl font-mono">
                  {(result.confidence * 100).toFixed(1)}%
                </span>
                <p className="text-xs text-zinc-500 mt-0.5">
                  confidence in {result.is_synthetic ? "synthetic" : "human"}
                </p>
              </div>
            </div>
            {/* Confidence bar */}
            <div className="mt-4 h-2 rounded-full bg-zinc-800 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${result.confidence * 100}%`,
                  backgroundColor: result.is_synthetic ? "#f87171" : "#34d399",
                }}
              />
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
