from dotenv import load_dotenv
load_dotenv()
import os
import uuid
import hashlib
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from schemas import BrandProfile, ReviewResponse, DismissalRequest, ApprovalAction
from engine import (
    extract_text_from_file,
    run_deterministic_checks,
    run_visual_and_accessibility_checks,
    run_ai_contextual_review,
)

app = FastAPI(title="BrandGuard AI API - Team Duocoders")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CURRENT_BRAND = BrandProfile()
SUBMISSIONS_DB = {}
AUDIT_LOGS = []
DISMISSAL_REASONS = []

@app.get("/api/brand", response_model=BrandProfile)
def get_brand_profile():
    return CURRENT_BRAND

@app.post("/api/brand/setup", response_model=BrandProfile)
def update_brand_profile(profile: BrandProfile):
    global CURRENT_BRAND
    CURRENT_BRAND = profile
    return CURRENT_BRAND

@app.post("/api/review/analyze", response_model=ReviewResponse)
async def analyze_submission(
    text_content: Optional[str] = Form(None),
    document: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    combined_text = text_content or ""
    if document:
        doc_bytes = await document.read()
        extracted = extract_text_from_file(doc_bytes, document.filename)
        combined_text = f"{combined_text}\n{extracted}".strip()

    if not combined_text and not image:
        raise HTTPException(status_code=400, detail="Provide text, document, or image.")

    findings = []
    if combined_text:
        findings.extend(run_deterministic_checks(combined_text, CURRENT_BRAND))
        findings.extend(run_ai_contextual_review(combined_text, CURRENT_BRAND, os.getenv("GEMINI_API_KEY")))

    if image:
        img_bytes = await image.read()
        findings.extend(run_visual_and_accessibility_checks(img_bytes, CURRENT_BRAND))

    penalty = sum(25 if f.severity == "high" else 12 if f.severity == "medium" else 5 for f in findings)
    score = max(0, 100 - penalty)
    version_hash = hashlib.sha256((combined_text + str(len(findings))).encode()).hexdigest()[:12]
    sub_id = f"sub-{uuid.uuid4().hex[:8]}"

    result = ReviewResponse(
        submission_id=sub_id,
        version_hash=version_hash,
        status="AI Reviewed",
        compliance_score=score,
        extracted_text=combined_text,
        findings=findings,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    SUBMISSIONS_DB[sub_id] = result.model_dump()
    return result

@app.post("/api/review/dismiss")
def dismiss_finding(req: DismissalRequest):
    if req.submission_id not in SUBMISSIONS_DB:
        raise HTTPException(status_code=404, detail="Submission not found")
    sub = SUBMISSIONS_DB[req.submission_id]
    sub["findings"] = [f for f in sub["findings"] if f["id"] != req.finding_id]
    DISMISSAL_REASONS.append({
        "finding_id": req.finding_id,
        "reason": req.reason,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    return {"status": "dismissed", "remaining_findings": len(sub["findings"])}

@app.post("/api/workflow/approve")
def reviewer_sign_off(action: ApprovalAction):
    if action.submission_id not in SUBMISSIONS_DB:
        raise HTTPException(status_code=404, detail="Submission not found")
    sub = SUBMISSIONS_DB[action.submission_id]
    if sub["version_hash"] != action.version_hash:
        raise HTTPException(status_code=409, detail="Version mismatch! Content was modified after scan.")
    sub["status"] = action.action
    event = {
        "submission_id": action.submission_id,
        "version_hash": action.version_hash,
        "reviewer": action.reviewer_name,
        "decision": action.action,
        "comment": action.comment,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    AUDIT_LOGS.append(event)
    return {"message": f"Submission locked as {action.action}", "audit_event": event}

@app.get("/api/analytics/insights")
def get_guideline_health():
    all_findings = [f for s in SUBMISSIONS_DB.values() for f in s["findings"]]
    by_category = {}
    for f in all_findings:
        by_category[f["category"]] = by_category.get(f["category"], 0) + 1
    return {
        "total_scans": len(SUBMISSIONS_DB),
        "active_violations_by_category": by_category,
        "dismissed_suggestions_count": len(DISMISSAL_REASONS),
        "guideline_ambiguity_alert": (
            "Rule clarification needed: Creators frequently dismiss tone flags on short social posts."
            if len(DISMISSAL_REASONS) >= 2 else "All guidelines operating with high clarity."
        ),
        "recent_audit_trail": AUDIT_LOGS[-5:]
    }