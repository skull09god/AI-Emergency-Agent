"use client";

import { useEffect, useState } from "react";

export default function AlertsPage() {
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState("timestamp");
  const [sortDir, setSortDir] = useState("desc"); // 'asc' | 'desc'
  const [page, setPage] = useState(1);
  const pageSize = 10;

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("http://127.0.0.1:8000/incidents");
        const data = await res.json();
        setIncidents(data);
      } catch (err) {
        console.error("Failed to load incidents", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const sorted = [...incidents].sort((a, b) => {
    let va = a[sortKey];
    let vb = b[sortKey];

    if (sortKey === "confidence") {
      va = Number(va);
      vb = Number(vb);
    }

    if (va < vb) return sortDir === "asc" ? -1 : 1;
    if (va > vb) return sortDir === "asc" ? 1 : -1;
    return 0;
  });

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const currentPage = Math.min(page, totalPages);
  const start = (currentPage - 1) * pageSize;
  const visible = sorted.slice(start, start + pageSize);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  if (loading) {
    return <p className="text-black">Loading alerts...</p>;
  }

  if (incidents.length === 0) {
    return (
      <div>
        <h1 className="text-black text-2xl font-bold mb-4">Alerts History</h1>
        <p className="text-black">No alerts yet.</p>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-black text-2xl font-bold mb-4">Alerts History</h1>
      <div className="flex items-center justify-between mb-2 text-black text-sm">
        <span>
          Sort by: <strong>{sortKey}</strong> ({sortDir})
        </span>
        <span>
          Page {currentPage} of {totalPages}
        </span>
      </div>
      <table className="min-w-full text-sm text-left text-black">
        <thead className="border-b">
          <tr>
            <th className="px-2 py-1 cursor-pointer" onClick={() => handleSort("id")}>
              ID
            </th>
            <th className="px-2 py-1 cursor-pointer" onClick={() => handleSort("timestamp")}>
              Time
            </th>
            <th className="px-2 py-1">File</th>
            <th className="px-2 py-1 cursor-pointer" onClick={() => handleSort("event")}>
              Event
            </th>
            <th className="px-2 py-1 cursor-pointer" onClick={() => handleSort("confidence")}>
              Conf
            </th>
            <th className="px-2 py-1 cursor-pointer" onClick={() => handleSort("status")}>
              Status
            </th>
          </tr>
        </thead>
        <tbody>
          {visible.map((i) => (
            <tr key={i.id} className="border-b">
              <td className="px-2 py-1">{i.id}</td>
              <td className="px-2 py-1">{i.timestamp}</td>
              <td className="px-2 py-1">{i.filename}</td>
              <td className="px-2 py-1">{i.event}</td>
              <td className="px-2 py-1">{Number(i.confidence).toFixed(2)}</td>
              <td className="px-2 py-1">{i.status}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="flex items-center gap-2 mt-3 text-black text-sm">
        <button
          className="px-2 py-1 border rounded disabled:opacity-50"
          onClick={() => setPage((p) => Math.max(1, p - 1))}
          disabled={currentPage === 1}
        >
          Prev
        </button>
        <button
          className="px-2 py-1 border rounded disabled:opacity-50"
          onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
          disabled={currentPage === totalPages}
        >
          Next
        </button>
      </div>
    </div>
  );
}
