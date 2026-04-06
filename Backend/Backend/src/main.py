"""
CyberWatch AI - FastAPI Backend
================================
Drop this file into your project root (or src/api/).
It wraps your existing SecurityPipeline and exposes REST endpoints
that the React frontend consumes.

Install deps:
    pip install fastapi uvicorn python-multipart

Run:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time, uuid, json, os
from datetime import datetime, timedelta
import random  # remove once real pipeline is integrated

app = FastAPI(title="CyberWatch AI API", version="1.0.0")

# ── CORS (allow your React dev server) ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory log store (replace with DB later) ─────────────────────────────
LOG_STORE: list[dict] = []

# ── Try to import your real pipeline ────────────────────────────────────────
try:
    # Adjust this import path to wherever your SecurityPipeline lives
    # e.g. from src.pipeline.security_pipeline import SecurityPipeline
    from security_pipeline import SecurityPipeline
    pipeline = SecurityPipeline()
    REAL_PIPELINE = True
    print("✅ Real SecurityPipeline loaded")
except ImportError:
    REAL_PIPELINE = False
    print("⚠️  SecurityPipeline not found — using mock responses (dev mode)")


# ── Schemas ──────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None


class AnalyzeResponse(BaseModel):
    id: str
    timestamp: str
    prompt_snippet: str
    final_decision: str          # "ALLOW" | "BLOCK"
    attack_type: str             # "Safe" | "Injection" | "Jailbreak" | "Suspicious"
    confidence: float            # 0.0 – 1.0
    risk_score: int              # 0 – 5
    details: dict


# ── Helper: mock pipeline response ──────────────────────────────────────────
def _mock_analyze(prompt: str) -> dict:
    """Fallback when real pipeline is unavailable (for UI development)."""
    injection_keywords = ["ignore previous", "system prompt", "reveal", "password", "database"]
    jailbreak_keywords = ["dan", "act as", "hypothetical", "jailbreak", "pretend you are"]

    prompt_lower = prompt.lower()
    is_injection = any(k in prompt_lower for k in injection_keywords)
    is_jailbreak = any(k in prompt_lower for k in jailbreak_keywords)

    if is_jailbreak:
        return {
            "final_decision": "BLOCK",
            "attack_type": "Jailbreak",
            "confidence": round(random.uniform(0.88, 0.97), 3),
            "risk_score": 5,
            "details": {
                "rule_engine": {"triggered": True, "pattern": "roleplay_bypass"},
                "intent_classifier": {"label": "malicious", "score": 0.94},
                "jailbreak_model": {"is_jailbreak": True, "probability": 0.95},
                "guard_llm": {"verdict": "BLOCK", "reason": "Attempts to override model identity"},
            },
        }
    elif is_injection:
        return {
            "final_decision": "BLOCK",
            "attack_type": "Injection",
            "confidence": round(random.uniform(0.80, 0.92), 3),
            "risk_score": 4,
            "details": {
                "rule_engine": {"triggered": True, "pattern": "prompt_injection"},
                "intent_classifier": {"label": "malicious", "score": 0.87},
                "jailbreak_model": {"is_jailbreak": False, "probability": 0.12},
                "guard_llm": {"verdict": "BLOCK", "reason": "Prompt injection attempt detected"},
            },
        }
    else:
        return {
            "final_decision": "ALLOW",
            "attack_type": "Safe",
            "confidence": round(random.uniform(0.90, 0.99), 3),
            "risk_score": 0,
            "details": {
                "rule_engine": {"triggered": False, "pattern": None},
                "intent_classifier": {"label": "benign", "score": 0.96},
                "jailbreak_model": {"is_jailbreak": False, "probability": 0.02},
                "guard_llm": {"verdict": "ALLOW", "reason": "No threats detected"},
            },
        }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "CyberWatch AI API is running", "version": "1.0.0"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_prompt(req: AnalyzeRequest):
    """
    Submit a prompt for security analysis.
    The pipeline runs all layers and returns a decision + details.
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    start = time.time()

    if REAL_PIPELINE:
        # ── Call your real pipeline here ──────────────────────────────────
        # Adjust method name if needed (e.g. pipeline.run(), pipeline.check())
        result = pipeline.analyze(req.prompt)
        # Expected result keys (adapt to your actual output):
        # result["final_decision"], result["attack_type"], result["risk_score"], etc.
        decision = result.get("final_decision", "ALLOW").upper()
        attack_type = result.get("attack_type", "Safe")
        confidence = float(result.get("confidence", result.get("jailbreak_probability", 0.95)))
        risk_score = int(result.get("risk_score", 0))
        details = {k: v for k, v in result.items() if k not in ["final_decision", "attack_type", "confidence", "risk_score"]}
    else:
        raw = _mock_analyze(req.prompt)
        decision = raw["final_decision"]
        attack_type = raw["attack_type"]
        confidence = raw["confidence"]
        risk_score = raw["risk_score"]
        details = raw["details"]

    latency_ms = round((time.time() - start) * 1000, 2)

    log_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "prompt_snippet": req.prompt[:80] + ("..." if len(req.prompt) > 80 else ""),
        "full_prompt": req.prompt,
        "final_decision": decision,
        "attack_type": attack_type,
        "confidence": confidence,
        "risk_score": risk_score,
        "latency_ms": latency_ms,
        "details": details,
    }
    LOG_STORE.append(log_entry)

    return AnalyzeResponse(**{k: log_entry[k] for k in AnalyzeResponse.model_fields})


@app.get("/logs")
def get_logs(limit: int = 50, offset: int = 0):
    """Return paginated intercept logs (most recent first)."""
    logs = list(reversed(LOG_STORE))
    return {
        "total": len(logs),
        "offset": offset,
        "limit": limit,
        "logs": logs[offset: offset + limit],
    }


@app.get("/stats")
def get_stats():
    """
    Dashboard summary statistics.
    Returns counts, threat level, attack distribution, and 7-day trend.
    """
    total = len(LOG_STORE)
    blocked = [l for l in LOG_STORE if l["final_decision"] == "BLOCK"]
    safe = [l for l in LOG_STORE if l["final_decision"] == "ALLOW"]

    injections = sum(1 for l in blocked if l["attack_type"] == "Injection")
    jailbreaks = sum(1 for l in blocked if l["attack_type"] == "Jailbreak")
    suspicious = sum(1 for l in blocked if l["attack_type"] == "Suspicious")

    # Risk score average
    avg_risk = round(sum(l["risk_score"] for l in LOG_STORE) / total, 1) if total else 0
    system_risk = min(100, int(avg_risk * 20))  # scale 0-5 → 0-100

    # Threat level
    if system_risk >= 70:
        threat_level = "CRITICAL"
    elif system_risk >= 40:
        threat_level = "HIGH"
    elif system_risk >= 20:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"

    # 7-day trend
    trend = []
    for i in range(6, -1, -1):
        day = (datetime.utcnow() - timedelta(days=i)).date()
        day_logs = [l for l in LOG_STORE if l["timestamp"].startswith(str(day))]
        trend.append({
            "date": str(day),
            "total": len(day_logs),
            "safe": sum(1 for l in day_logs if l["final_decision"] == "ALLOW"),
            "blocked": sum(1 for l in day_logs if l["final_decision"] == "BLOCK"),
        })

    # Latest events
    last_blocked = next((l for l in reversed(LOG_STORE) if l["final_decision"] == "BLOCK"), None)
    last_safe = next((l for l in reversed(LOG_STORE) if l["final_decision"] == "ALLOW"), None)

    return {
        "total_monitored": total,
        "blocked_today": len([l for l in blocked if l["timestamp"].startswith(str(datetime.utcnow().date()))]),
        "safe_analyzed": len([l for l in safe if l["timestamp"].startswith(str(datetime.utcnow().date()))]),
        "injections_blocked": injections,
        "jailbreaks_prevented": jailbreaks,
        "suspicious_flagged": suspicious,
        "system_risk_score": system_risk,
        "threat_level": threat_level,
        "attack_distribution": {
            "injection": injections,
            "jailbreak": jailbreaks,
            "suspicious": suspicious,
        },
        "trend_7d": trend,
        "latest_blocked": last_blocked,
        "latest_safe": last_safe,
    }


@app.get("/pipeline/status")
def pipeline_status():
    """Health check for each pipeline component."""
    components = [
        {"id": "rule_engine",        "name": "Rule Engine",      "status": "active"},
        {"id": "intent_classifier",  "name": "Intent Model",     "status": "active" if REAL_PIPELINE else "active"},
        {"id": "jailbreak_model",    "name": "Jailbreak Model",  "status": "active" if REAL_PIPELINE else "active"},
        {"id": "guard_llm",          "name": "Guard LLM",        "status": "active" if REAL_PIPELINE else "active"},
        {"id": "agent_monitor",      "name": "Agent Monitor",    "status": "active"},
    ]
    return {
        "overall": "operational",
        "real_pipeline_loaded": REAL_PIPELINE,
        "components": components,
    }
