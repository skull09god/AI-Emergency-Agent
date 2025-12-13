# AI Emergency Agent

## 1. Overview

AI Emergency Agent is an AI‑powered emergency detection and escalation system built to analyze short CCTV‑style videos and identify potentially dangerous situations in near real time. The system ingests video clips, extracts representative frames, applies machine‑learning models to detect incidents such as fights, threats, thefts, or abnormal activity, assigns a confidence and risk level, logs the incident, and can notify responders automatically. The project consists of a FastAPI backend, a Next.js web dashboard**, multiple ML pipelines for video and image understanding, and clearly defined integration points for sponsor technologies such as Cline, Kastra, Oumi, Vercel, and Coderabbit.

---

## 2. Problem and Solution

### The Problem

Emergency situations in public or private spaces (fights, threats, thefts, accidents) are often detected too late. Traditional CCTV systems rely on manual monitoring, which is expensive, error‑prone, and does not scale well across hundreds or thousands of camera feeds. As a result, critical incidents may go unnoticed or unaddressed until damage has already occurred.

### The Solution

AI Emergency Agent introduces an end‑to‑end AI pipeline that automatically processes video clips, extracts frames, classifies the situation (normal vs. emergency categories), computes a confidence score, and decides whether the incident should be escalated. This reduces dependence on continuous human monitoring and enables faster, more consistent responses.

The system also provides a web dashboard where operators can upload videos, review detected incidents, and inspect historical alerts. Integrations with messaging (Telegram) and designed orchestration workflows (Kastra), along with AI agents and tooling (Oumi, Cline, Coderabbit), improve reliability, development speed, and decision support under hackathon constraints.

---

## 3. Key Features

* **Video upload & analysis**
  Upload short video clips via the dashboard. The backend extracts frames and runs AI analysis on them.

* **Video frame extraction**
  Utilities and scripts to extract frames from videos for dataset creation, offline inspection, and model training.

* **Image & motion classification**

  * **ResNet‑based image classifier** to label frames into emergency classes such as `normal`, `fight`, `theft`, or `threat`.
  * **Motion‑based detector** using optical flow features and a lightweight classifier to estimate activity intensity.

* **Incident inference endpoint** (`/infer-incident`)

  * Runs ResNet‑based video analysis across sampled frames.
  * Computes a confidence score.
  * Assigns `high_risk` if the event is not `normal` and confidence exceeds a threshold; otherwise assigns `low_risk`.
  * Logs each incident in memory with: `id`, timestamp (converted to IST), filename, event, confidence, number of frames analyzed, and status.

* **Alerts dashboard**

  * Next.js dashboard with pages for video upload and incident history.
  * Incident history is sortable and paginated.
  * Uses `NEXT_PUBLIC_API_BASE_URL` to communicate with the backend in both local and deployed environments.

* **Real‑time notifications**

  * When a `high_risk` incident is detected, the backend can send a Telegram alert containing event type, confidence, file name, and IST timestamp.

* **Model training utilities**

  * Scripts for training an emergency image classifier (e.g., ResNet18) from labeled image datasets.
  * Scripts to generate JSONL datasets and train a vision‑language model based on **Qwen2‑VL‑2B‑Instruct** for richer scene understanding.

* **Reinforcement learning policy (Oumi)**

  * Training script using **Oumi** to learn escalation policies (e.g., alert police vs. monitor) from incident prompts and outcomes.

* **API endpoints**

  * `/classify-frame` – classify a single image frame.
  * `/infer-video-resnet` – analyze a video using ResNet across sampled frames.
  * `/infer-video` – motion‑based video analysis using optical flow.
  * `/infer-incident` – wrap model output into an incident with a risk status.
  * `/incidents` – list recent incidents.
  * `/escalate` – simple rule‑based escalation endpoint with a placeholder for Oumi‑driven decisions.

* **Sponsor‑friendly architecture**

  * Kastra workflow definition for “emergency detected” events.
  * Development workflow enhanced with Coderabbit (code review) and Cline (AI coding assistant).

---

## 4. Tech Stack

### Frontend

* React
* Next.js (App Router)
* Tailwind CSS / basic CSS for styling

### Backend

* FastAPI (Python)
* Uvicorn (ASGI server)

### Video & Image Processing

* OpenCV (`cv2`) for frame extraction and optical flow
* NumPy for numerical operations

### Machine Learning

* PyTorch and `torchvision` (ResNet‑based models)
* scikit‑learn and joblib for motion‑based detection pipelines
* Qwen2‑VL‑2B‑Instruct for vision‑language understanding
* Oumi for reinforcement‑learning policy training

### Messaging & Integrations

* Telegram Bot API (via `httpx`)
* Kastra/Kestra workflow YAML for incident orchestration

### Data & Configuration

* Hugging Face datasets (where applicable)
* JSON and YAML configuration files

### Development & Tooling

* Poetry or pip for Python dependency management
* npm for frontend dependencies
* GitHub for source control
* Coderabbit and Cline for AI‑assisted development
* Vercel for frontend hosting
* Render (or similar) for backend deployment

---

## 5. Sponsor Technologies

### Vercel

* Used to deploy the Next.js dashboard.
* Environment variables such as `NEXT_PUBLIC_API_BASE_URL` allow seamless switching between local and production backends.
* Automatic deployments on git push and preview URLs enable fast UI iteration during the hackathon.

### Coderabbit

* Installed as a GitHub App on the repository to perform AI‑powered code reviews on pull requests.
* Reviewed changes to critical endpoints like `/infer-incident`, dashboard data‑fetching logic, and refactors.
* Helped maintain code quality and catch issues under tight hackathon timelines.

### Cline

* Used as an AI coding assistant inside VS Code.
* Assisted with drafting and refactoring FastAPI endpoints.
* Helped generate and refine TypeScript/React components for the dashboard.
* Suggested tests and documentation snippets.
* Integrated into the development workflow (not part of the runtime system).

### Kastra (Design Integration)

* Defined a Kastra/Kestra workflow named `emergency-incident-workflow` in the namespace `ai-emergency-dashboard`.
* Workflow inputs include `event`, `confidence`, and `clip_url`.
* The workflow fans out via HTTP tasks such as `notify-user` and `trigger-cline` to automate responses when a high‑risk incident is detected.
* The backend includes a `KASTRA_FLOW_URL` environment variable and a TODO hook showing where the workflow would be triggered after `/infer-incident`.

### Oumi

* Used conceptually to train an RL policy that determines escalation strategies based on incident context.
* The `/escalate` endpoint returns a decision (e.g., `ALERT_POLICE` vs. `MONITOR`) and includes an `oumi_used` flag / TODO placeholder.
* Positioned as the reasoning layer on top of raw detection models.



---

## 6. Getting Started / Setup

### Prerequisites

* Python 3.9+
* Node.js 18+
* Poetry or pip
* npm
* Optional: CUDA‑enabled GPU for faster training
* Telegram Bot Token and Chat ID (for alerts)

### Clone the Repository

```bash
git clone https://github.com/skull09god/AI-Emergency-Agent.git
cd AI-Emergency-Agent
```

### Backend Setup

```bash
cd backend
poetry install  # or pip install -r requirements.txt
```



### Frontend Setup

```bash
cd dashboard
npm install
```

Set environment variables:

```env
# For local development
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# For production (Vercel)
# This should point to your deployed backend URL
NEXT_PUBLIC_API_BASE_URL=https://ai-emergency-agent.onrender.com/
```

### Detection Model Training (Optional)

```bash
cd detection
python train_model.py
```

### Running Locally

Backend:

```bash
# Assumes backend/app.py contains: app = FastAPI()
uvicorn app:app --reload
```

Frontend:

```bash
npm run dev
```

Open: [http://localhost:3000](http://localhost:3000)

---

## 7. Usage

**Deployed URLs**

* Frontend (Vercel): [https://ai-emergency-agent-z468.vercel.app/](https://ai-emergency-agent-z468.vercel.app/)
* Backend (Render): [https://ai-emergency-agent.onrender.com/](https://ai-emergency-agent.onrender.com/)

1. Upload a video via the dashboard.
2. The backend extracts frames and runs analysis.
3. View model output (event, confidence, frames analyzed) and incident status (`high_risk` / `low_risk`).
4. When `high_risk`:

   * A Telegram alert is sent .
   * The incident appears at the top of the Alerts history.
   * A Kastra workflow would be triggered.
   * Oumi would generate escalation recommendations.

---

## 8. Project Structure

```
AI-Emergency-Agent/
├── backend/              # FastAPI backend (app.py, API routes, ML inference logic)
│   ├── app.py            # FastAPI entry point (app = FastAPI())
│   ├── models/           # Loaded ML models (ResNet, motion detector)
│   ├── services/         # Telegram alerts, Kastra hooks, escalation logic
│   └── utils/            # Video/frame extraction, helpers
├── dashboard/            # Next.js frontend dashboard
│   ├── app/              # App Router pages (upload, alerts)
│   ├── components/       # UI components (e.g., UploadVideo, IncidentTable)
│   └── lib/              # API helpers and config
├── detection/            # ML training and dataset utilities
│   ├── train_model.py    # ResNet-based emergency classifier training
│   ├── motion_detector/  # Optical flow + activity detection pipeline
│   └── vlm/              # Qwen2-VL dataset prep and training scripts
├── workflows/            # Kastra/Kestra workflow definitions
│   └── emergency-incident-workflow.yaml
├── README.md             # Project documentation
```

---

## 9. Screenshots / Demo

* Dashboard upload page <img width="878" height="617" alt="Screenshot 2025-12-13 170534" src="https://github.com/user-attachments/assets/943d86e0-0ebe-4f9b-beed-69ab238a1b11" />

* Alerts history view <img width="1907" height="366" alt="Screenshot 2025-12-13 170550" src="https://github.com/user-attachments/assets/96be3ba1-cd39-4f00-8af1-13021347d405" />

* Telegram alert <img width="540" height="1200" alt="image" src="https://github.com/user-attachments/assets/7bef03e3-e345-4652-a7a7-d29dc20bc373" />
* Coderabbit PR review or Cline usage

---

## 10. Contributing, License, Contact

### Contributing

Contributions are welcome. Please open an issue or submit a pull request with a clear description of changes.

### License

This project is licensed under the MIT License.

### Contact

For questions or feedback, please use GitHub Issues or contact the project team via the repository.
email:pranjalshahi1920@gmail.com
