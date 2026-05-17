# SHL Conversational Assessment Recommender

FastAPI service that recommends SHL Individual Test Solutions through a
multi-turn conversation with a hiring manager or recruiter.

---

## Setup & Run (Local)

### Step 1 — Install dependencies
```bash
cd shl-recommender
pip install -r requirements.txt
```

### Step 2 — Get a free Gemini API key
https://aistudio.google.com/app/apikey  (takes 30 seconds, no credit card)

### Step 3 — Set your API key
```bash
# Mac / Linux:
export GEMINI_API_KEY=paste_your_key_here

# Windows PowerShell:
$env:GEMINI_API_KEY="paste_your_key_here"
```

### Step 4 — Start the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 5 — Test it
```bash
# Health check:
curl http://localhost:8000/health

# Quick chat test:
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer who works with stakeholders."}]}'

# Full test suite (open a second terminal, server must be running):
python scripts/test_agent.py
```

---

## Deploy to Render (Free — required for submission)

1. Push this folder to a new GitHub repo.
2. Go to https://render.com → **New Web Service** → connect the repo.
3. Render auto-reads `render.yaml` and configures everything.
4. In the Render dashboard → **Environment** tab → add:
   - Key: `GEMINI_API_KEY`
   - Value: your key from Step 2
5. Click **Deploy**.
6. Your public endpoint: `https://<service-name>.onrender.com`
7. Submit `https://<service-name>.onrender.com` as your API endpoint URL.

> **Note on cold start:** Render free tier sleeps after 15 min of inactivity.
> The evaluator allows up to 2 minutes for the first `/health` call — this is fine.

---

## API Reference

### GET /health
```json
{"status": "ok"}
```

### POST /chat
**Request:**
```json
{
  "messages": [
    {"role": "user",      "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user",      "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments for a mid-level Java developer with stakeholder interaction needs.",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r",                           "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` = `[]` while clarifying or refusing.
- `end_of_conversation` = `true` only when the user confirms a final shortlist.

---

## Project Structure
```
shl-recommender/
├── main.py              ← FastAPI app + agent logic
├── requirements.txt
├── render.yaml          ← Render deployment config
├── .env.example
├── data/
│   └── catalog.json     ← 81 SHL Individual Test Solutions (from official JSON)
├── scripts/
│   └── test_agent.py    ← 10 test cases (modelled on sample conversations)
├── README.md
└── APPROACH.md          ← 2-page design doc for submission
```

---

## Tech Stack
| Component | Technology |
|-----------|-----------|
| API | FastAPI + Uvicorn |
| LLM | Google Gemini 1.5 Flash (free) |
| Catalog | Official SHL JSON (81 assessments) |
| Deployment | Render.com (free tier) |
| Validation | Pydantic v2 |
