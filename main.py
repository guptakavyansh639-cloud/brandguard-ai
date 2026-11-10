import os
import re
import time
import uuid
import base64
import hashlib
from collections import deque
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional: Uses Gemini 2.5 Flash if GEMINI_API_KEY is set, otherwise runs fast local fallback
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = bool(os.getenv("GEMINI_API_KEY"))
    client = genai.Client() if GEMINI_AVAILABLE else None
except ImportError:
    GEMINI_AVAILABLE = False
    client = None

app = FastAPI(
    title="BrandGuard AI Enterprise Gateway",
    version="2.5.0"
)

# Enable CORS so your Vercel frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STEP 3: In-Memory Live Audit Trail Storage ---
AUDIT_LOGS = deque(maxlen=20)

# --- STEP 1: Pydantic Request & Gemini Structured Output Schemas ---
class AnalyzeRequest(BaseModel):
    text: Optional[str] = ""
    prompt: Optional[str] = ""
    content: Optional[str] = ""
    brand_rules: Optional[str] = "Tone: Professional & Empathetic. Never mention rival AcmeCorp. No 100% guaranteed financial returns."
    image_base64: Optional[str] = None
    modules: Optional[List[str]] = ["pii", "toxicity", "brand", "injection", "copyright"]

class GeminiStructuredEval(BaseModel):
    risk_score: int = Field(description="0 to 100 risk score")
    violations: List[str] = Field(description="List of specific policy or brand violations")
    sanitized_rewrite: str = Field(description="Production-ready safe rewrite with PII redacted and brand tone fixed")
    brand_alignment_score: int = Field(description="0 to 100 score on how well the text fits the brand rules")
    explanation: str = Field(description="1-sentence explanation of why this score was given")

def stage1_fast_regex_scan(text: str) -> Dict[str, Any]:
    """Stage 1: Sub-5ms deterministic PII, Secret & Competitor scanner"""
    violations = []
    sanitized = text

    if re.search(r'[\w\.-]+@[\w\.-]+\.\w+', sanitized):
        violations.append("PII_LEAK: Email Address")
        sanitized = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', sanitized)

    if re.search(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', sanitized):
        violations.append("PII_LEAK: Phone Number")
        sanitized = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]', sanitized)

    if re.search(r'(sk-[a-zA-Z0-9]{16,}|AIza[0-9A-Za-z-_]{35}|\b\d{3}-\d{2}-\d{4}\b)', sanitized):
        violations.append("SECURITY: Exposed Secret / SSN")
        sanitized = re.sub(r'(sk-[a-zA-Z0-9]{16,}|AIza[0-9A-Za-z-_]{35}|\b\d{3}-\d{2}-\d{4}\b)', '[REDACTED_SECRET]', sanitized)

    if re.search(r'(ignore (all )?previous instructions|system prompt|bypass|jailbreak|DAN mode)', sanitized, re.I):
        violations.append("SECURITY: Prompt Injection Attempt")

    if re.search(r'(AcmeCorp|AcmeAI)', sanitized, re.I):
        violations.append("BRAND_VIOLATION: Competitor Mention (AcmeCorp)")
        sanitized = re.sub(r'(AcmeCorp|AcmeAI)', '[COMPETITOR_OMITTED]', sanitized, flags=re.I)

    if re.search(r'(100%\s*guaranteed|risk[- ]free return|double your money)', sanitized, re.I):
        violations.append("COMPLIANCE: Unverified Financial Guarantee")
        sanitized = re.sub(r'(100%\s*guaranteed|risk[- ]free return|double your money)', 'historically strong performance (subject to terms)', sanitized, flags=re.I)

    if re.search(r'\b(idiot|stupid|scam|garbage|hate you|useless|moron)\b', sanitized, re.I):
        violations.append("SAFETY: Toxic / Off-Brand Hostility")
        sanitized = "We understand your concern and are committed to resolving this professionally. Please contact our support team."

    return {"violations": violations, "sanitized_text": sanitized}

# --- STEP 2: Main Multi-Modal + Hybrid Guardrail Endpoint ---
@app.post("/api/review/analyze")
async def analyze_guardrail(req: AnalyzeRequest):
    t_start = time.perf_counter()

    # Support whichever field name the frontend sends (text, prompt, or content)
    raw_text = req.text or req.prompt or req.content or ""

    # Stage 1: Local Regex Scan
    t0 = time.perf_counter()
    s1 = stage1_fast_regex_scan(raw_text)
    stage1_ms = round((time.perf_counter() - t0) * 1000, 2)

    violations = list(s1["violations"])
    sanitized_output = s1["sanitized_text"]
    risk_score = min(100, len(violations) * 35)
    explanation = "Evaluated via Stage-1 Deterministic Regex + Heuristic Guardrail."
    brand_score = max(0, 100 - risk_score)

    # Stage 2: Gemini 2.5 Flash (Text + Optional Base64 Image)
    stage2_ms = 0.0
    if GEMINI_AVAILABLE and (raw_text.strip() or req.image_base64):
        t1 = time.perf_counter()
        try:
            contents = [
                f"""You are BrandGuard AI. Analyze the submission against safety policies and these Brand Rules: {req.brand_rules}.
                Input Text: "{raw_text}"
                Stage-1 Pre-Redacted Text: "{sanitized_output}"
                Stage-1 Detected Violations: {violations}
                If an image is attached, also inspect the image for competitor logos, copyright infringement, or unsafe visuals.
                Return JSON with risk_score (0-100), complete violations list, sanitized_rewrite (safe, on-brand copy ready for production), brand_alignment_score, and explanation."""
            ]

            if req.image_base64:
                raw_b64 = req.image_base64.split(",")[-1]
                img_bytes = base64.b64decode(raw_b64)
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiStructuredEval,
                    temperature=0.1,
                ),
            )
            llm_data = GeminiStructuredEval.model_validate_json(resp.text)
            risk_score = max(risk_score, llm_data.risk_score)
            violations = list(dict.fromkeys(violations + llm_data.violations))
            sanitized_output = llm_data.sanitized_rewrite
            brand_score = llm_data.brand_alignment_score
            explanation = llm_data.explanation
        except Exception as e:
            explanation = f"Stage-1 Guardrail enforced (Stage-2 fallback: {str(e)[:40]})"
        stage2_ms = round((time.perf_counter() - t1) * 1000, 2)

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)

    if risk_score >= 70:
        risk_level, status, action = "CRITICAL", "BLOCKED", "BLOCKED"
    elif risk_score >= 30:
        risk_level, status, action = "MODERATE", "FLAGGED", "AUTO_SANITIZED"
    else:
        risk_level, status, action = "LOW", "APPROVED", "ALLOWED"

    submission_id = f"bg-{uuid.uuid4().hex[:8]}"
    audit_hash = hashlib.sha256(f"{raw_text}:{risk_score}:{time.time()}".encode()).hexdigest()[:16]

    # Record in Live Audit Trail (Step 3)
    AUDIT_LOGS.appendleft({
        "time": datetime.utcnow().strftime("%H:%M:%S UTC"),
        "submission_id": submission_id,
        "text_snippet": raw_text[:45] + ("..." if len(raw_text) > 45 else ""),
        "modules": req.modules,
        "risk_score": risk_score,
        "action": action
    })

    return {
        "submission_id": submission_id,
        "status": status,
        "action": action,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "violations": violations,
        "detected_violations": violations,
        "sanitized_rewrite": sanitized_output,
        "explanation": explanation,
        "auto_remediation": {
            "action_taken": "REDACTED_AND_REWRITTEN" if violations else "NONE_REQUIRED",
            "original_input": raw_text,
            "production_safe_output": sanitized_output,
            "brand_alignment_score": f"{brand_score}/100"
        },
        "telemetry": {
            "stage1_regex_ms": stage1_ms,
            "stage2_gemini_ms": stage2_ms,
            "total_ms": total_ms
        },
        "sha256_audit_signature": audit_hash,
        "latency_ms": total_ms
    }

# --- STEP 3: Endpoint to Fetch Live Audit Trail ---
@app.get("/api/review/logs")
async def get_audit_logs():
    return {"logs": list(AUDIT_LOGS)}

@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "Gemini 2.5 Flash Hybrid", "gemini_active": GEMINI_AVAILABLE}