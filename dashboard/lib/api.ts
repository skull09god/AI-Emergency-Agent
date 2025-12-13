const BACKEND_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

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
