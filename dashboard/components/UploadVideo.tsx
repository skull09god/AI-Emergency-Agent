"use client";

import { useState } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

export default function UploadVideo() {
  const [file, setFile] = useState(null);
  const [loadingAnalyze, setLoadingAnalyze] = useState(false);
  const [loadingIncident, setLoadingIncident] = useState(false);
  const [result, setResult] = useState(null);
  const [incident, setIncident] = useState(null);
  const [error, setError] = useState(null);

  function handleFileChange(e) {
    const f = e.target.files?.[0] || null;
    setFile(f);
    setResult(null);
    setIncident(null);
    setError(null);
  }

  async function callBackend(path, setState, setLoading) {
    if (!file) {
      setError("Please select a video first.");
      return;
    }
    setLoading(true);
    setError(null);
    setState(null);

    try {
      const formData = new FormData();
      formData.append("video", file);

      const res = await fetch(`${BACKEND_URL}${path}`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Backend error");
      }

      const data = await res.json();
      setState(data);
    } catch (err) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleAnalyze() {
    await callBackend("/infer-video-resnet", setResult, setLoadingAnalyze);
  }

  async function handleLogIncident() {
    await callBackend("/infer-incident", setIncident, setLoadingIncident);
  }

  return (
    <div className="space-y-4">
      <input
        type="file"
        accept="video/*"
        onChange={handleFileChange}
        className="text-black"
      />

      <div className="flex gap-2">
        <button
          onClick={handleAnalyze}
          disabled={loadingAnalyze || !file}
          className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-gray-400"
        >
          {loadingAnalyze ? "Analyzing..." : "Upload & Analyze"}
        </button>

        <button
          onClick={handleLogIncident}
          disabled={loadingIncident || !file}
          className="px-4 py-2 bg-green-600 text-white rounded disabled:bg-gray-400"
        >
          {loadingIncident ? "Logging..." : "Log Incident"}
        </button>
      </div>

      {error && <p className="text-red-600 mt-2">{error}</p>}

      {result && (
        <div className="mt-4 text-black">
          <p><strong>Event:</strong> {result.event}</p>
          <p><strong>Confidence:</strong> {result.confidence.toFixed(2)}</p>
          <p><strong>Frames analyzed:</strong> {result.frames_analyzed}</p>
          <p><strong>ResNet frames:</strong> {result.resnet_frames}</p>
        </div>
      )}

      {incident && (
        <div className="mt-4 text-black border-t pt-2">
          <p><strong>Incident logged.</strong></p>
          <p><strong>Event:</strong> {incident.event}</p>
          <p><strong>Confidence:</strong> {incident.confidence.toFixed(2)}</p>
          <p><strong>Status:</strong> {incident.status}</p>
          <p><strong>Frames analyzed:</strong> {incident.frames_analyzed}</p>
        </div>
      )}
    </div>
  );
}
