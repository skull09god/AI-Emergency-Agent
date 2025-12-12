const BACKEND_URL = "http://127.0.0.1:8000";

export async function inferIncident(videoFile: File) {
  const formData = new FormData();
  formData.append("video", videoFile);

  const res = await fetch(`${BACKEND_URL}/infer-incident`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchIncidents() {
  const res = await fetch(`${BACKEND_URL}/incidents?limit=50`);
  if (!res.ok) throw new Error("Failed to fetch incidents");
  return res.json();
}
