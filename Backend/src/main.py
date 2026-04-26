"""
main.py — CyberWatch AI FastAPI backend
========================================
Fixed version. Changes:
  - Reads max_risk from pipeline result (fixes Risk 6/5 display)
  - Reads real confidence instead of defaulting to 0.95
  - Passes attack_type as-is (now has 8 categories from pipeline)
  - risk_score sent as raw value; frontend should display as X / max_risk
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time, uuid, os
from datetime import datetime, timedelta
import random

app = FastAPI(title="CyberWatch AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_STORE: list[dict] = []

try:
    from security_pipeline import SecurityPipeline
    pipeline = SecurityPipeline()
    REAL_PIPELINE = True
    print("✅ Real SecurityPipeline loaded")
except ImportError as e:
    REAL_PIPELINE = False
    print(f"⚠️  SecurityPipeline not found — using mock responses. Error: {e}")


# ── Schemas ───────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None


class AnalyzeResponse(BaseModel):
    id:             str
    timestamp:      str
    prompt_snippet: str
    final_decision: str
    attack_type:    str
    confidence:     float
    risk_score:     int
    max_risk:       int       # ← NEW: frontend shows risk_score/max_risk
    details:        dict


# ── Mock pipeline (dev mode only) ─────────────────────────────────────────────
def _mock_analyze(prompt: str) -> dict:
    p = prompt.lower()
    FINGERPRINTS = {
        "ROLEPLAY_JAILBREAK":    ["act as", "pretend you are", "play the role"],
        "PERSONA_SWAP":          ["no restrictions", "no guidelines", "no limits"],
        "HYPOTHETICAL_BYPASS":   ["hypothetically", "imagine you could"],
        "POLICY_BYPASS":         ["ignore your policy", "bypass your filter"],
        "SYSTEM_PROMPT_EXTRACTION": ["reveal your instructions", "show your prompt"],
        "INSTRUCTION_OVERRIDE":  ["ignore previous", "disregard", "forget your"],
        "TRAINING_MODE_EXPLOIT": ["developer mode", "debug mode", "training mode"],
        "INDIRECT_INJECTION":    ["the document says", "according to the context"],
    }
    attack_type = "SAFE"
    for cat, fps in FINGERPRINTS.items():
        if any(fp in p for fp in fps):
            attack_type = cat
            break
    if attack_type == "SAFE" and any(k in p for k in ["dan", "jailbreak"]):
        attack_type = "JAILBREAK"

    is_attack = attack_type != "SAFE"
    return {
        "final_decision":  "BLOCK" if is_attack else "ALLOW",
        "attack_type":     attack_type,
        "confidence":      round(random.uniform(0.82, 0.97), 3) if is_attack else round(random.uniform(0.90, 0.99), 3),
        "risk_score":      random.randint(3, 7) if is_attack else random.randint(0, 1),
        "max_risk":        8,
        "rule_engine":     {"hits": [], "decision": "SUSPICIOUS" if is_attack else "SAFE"},
        "intent_classifier": {"intent": "malicious" if is_attack else "benign",
                              "confidence": 0.91, "decision": "MALICIOUS" if is_attack else "SAFE"},
        "jailbreak_detector": {"probability": 0.89 if is_attack else 0.03,
                               "decision": "JAILBREAK DETECTED" if is_attack else "SAFE"},
        "guard_llm":       {"decision": "BLOCK" if is_attack else "SAFE", "is_blocked": is_attack},
        "preprocessor":    {"detected_attacks": [], "risk_boost": 0,
                            "was_modified": False, "flags": {}, "attack_category": ""},
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "CyberWatch AI API is running", "version": "1.0.0"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_prompt(req: AnalyzeRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    start = time.time()

    if REAL_PIPELINE:
        result = pipeline.analyze(req.prompt)

        decision    = result.get("final_decision", "ALLOW").upper()
        attack_type = result.get("attack_type", "SAFE")

        # BUG 2 FIX: use real computed confidence, not a hardcoded 0.95
        confidence  = float(result.get("confidence", 0.5))

        # BUG 1 FIX: read max_risk so frontend can show "4/8" not "4/5"
        risk_score  = int(result.get("risk_score", 0))
        max_risk    = int(result.get("max_risk", 8))

        # Everything except the top-level keys goes into details
        details = {
            k: v for k, v in result.items()
            if k not in ["final_decision", "attack_type",
                         "confidence", "risk_score", "max_risk", "timestamp"]
        }
    else:
        raw         = _mock_analyze(req.prompt)
        decision    = raw["final_decision"]
        attack_type = raw["attack_type"]
        confidence  = raw["confidence"]
        risk_score  = raw["risk_score"]
        max_risk    = raw["max_risk"]
        details     = {k: v for k, v in raw.items()
                       if k not in ["final_decision", "attack_type",
                                    "confidence", "risk_score", "max_risk"]}

    latency_ms = round((time.time() - start) * 1000, 2)

    log_entry = {
        "id":             str(uuid.uuid4()),
        "timestamp":      datetime.utcnow().isoformat() + "Z",
        "prompt_snippet": req.prompt[:80] + ("..." if len(req.prompt) > 80 else ""),
        "full_prompt":    req.prompt,
        "final_decision": decision,
        "attack_type":    attack_type,
        "confidence":     confidence,
        "risk_score":     risk_score,
        "max_risk":       max_risk,
        "latency_ms":     latency_ms,
        "details":        details,
    }
    LOG_STORE.append(log_entry)

    return AnalyzeResponse(**{k: log_entry[k] for k in AnalyzeResponse.model_fields})


@app.get("/logs")
def get_logs(limit: int = 50, offset: int = 0):
    logs = list(reversed(LOG_STORE))
    return {
        "total":  len(logs),
        "offset": offset,
        "limit":  limit,
        "logs":   logs[offset: offset + limit],
    }


@app.get("/stats")
def get_stats():
    total   = len(LOG_STORE)
    blocked = [l for l in LOG_STORE if l["final_decision"] == "BLOCK"]
    safe    = [l for l in LOG_STORE if l["final_decision"] == "ALLOW"]

    # BUG 3 FIX: count all attack categories, not just 2
    attack_counts = {}
    for l in blocked:
        cat = l.get("attack_type", "UNKNOWN")
        attack_counts[cat] = attack_counts.get(cat, 0) + 1

    # Keep backward-compat keys for dashboard
    injections = sum(v for k, v in attack_counts.items()
                     if "INJECTION" in k or "INDIRECT" in k or k == "PROMPT_INJECTION")
    jailbreaks = sum(v for k, v in attack_counts.items()
                     if "JAILBREAK" in k or "ROLEPLAY" in k or "PERSONA" in k
                     or "HYPOTHETICAL" in k or "TRAINING" in k)
    suspicious = sum(v for k, v in attack_counts.items()
                     if "SUSPICIOUS" in k or "POLICY" in k
                     or "OVERRIDE" in k or "SYSTEM_PROMPT" in k)

    avg_risk    = round(sum(l["risk_score"] for l in LOG_STORE) / total, 1) if total else 0
    max_r       = LOG_STORE[-1].get("max_risk", 8) if LOG_STORE else 8
    system_risk = min(100, int((avg_risk / max_r) * 100))

    if system_risk >= 70:
        threat_level = "CRITICAL"
    elif system_risk >= 40:
        threat_level = "HIGH"
    elif system_risk >= 20:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"

    trend = []
    for i in range(6, -1, -1):
        day      = (datetime.utcnow() - timedelta(days=i)).date()
        day_logs = [l for l in LOG_STORE if l["timestamp"].startswith(str(day))]
        trend.append({
            "date":    str(day),
            "total":   len(day_logs),
            "safe":    sum(1 for l in day_logs if l["final_decision"] == "ALLOW"),
            "blocked": sum(1 for l in day_logs if l["final_decision"] == "BLOCK"),
        })

    last_blocked = next((l for l in reversed(LOG_STORE) if l["final_decision"] == "BLOCK"), None)
    last_safe    = next((l for l in reversed(LOG_STORE) if l["final_decision"] == "ALLOW"), None)

    today = str(datetime.utcnow().date())
    return {
        "total_monitored":     total,
        "blocked_today":       len([l for l in blocked if l["timestamp"].startswith(today)]),
        "safe_analyzed":       len([l for l in safe    if l["timestamp"].startswith(today)]),
        "injections_blocked":  injections,
        "jailbreaks_prevented":jailbreaks,
        "suspicious_flagged":  suspicious,
        "system_risk_score":   system_risk,
        "threat_level":        threat_level,
        "attack_distribution": attack_counts,       # full breakdown now
        "trend_7d":            trend,
        "latest_blocked":      last_blocked,
        "latest_safe":         last_safe,
    }


@app.get("/pipeline/status")
def pipeline_status():
    components = [
        {"id": "preprocessor",      "name": "Preprocessor",    "status": "active"},
        {"id": "rule_engine",       "name": "Rule Engine",     "status": "active"},
        {"id": "intent_classifier", "name": "Intent Model",    "status": "active" if REAL_PIPELINE else "mock"},
        {"id": "jailbreak_model",   "name": "Jailbreak Model", "status": "active" if REAL_PIPELINE else "mock"},
        {"id": "guard_llm",         "name": "Guard LLM",       "status": "active" if REAL_PIPELINE else "mock"},
        {"id": "agent_monitor",     "name": "Agent Monitor",   "status": "active"},
    ]
    return {
        "overall":              "operational",
        "real_pipeline_loaded": REAL_PIPELINE,
        "components":           components,
    }