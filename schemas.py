from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class BrandProfile(BaseModel):
    brand_name: str = "NovaPay Fintech"
    approved_tone: str = "Clear, empowering, transparent, and professional. Never hype-driven."
    banned_terms: List[str] = ["100% guaranteed", "risk-free", "instant wealth", "cheapest"]
    required_disclaimer: str = "Subject to credit approval. Terms and conditions apply."
    official_facts: dict = {
        "max_cashback": "5%",
        "annual_fee": "$0",
        "support_hours": "24/7"
    }
    brand_hex_palette: List[str] = ["#0F172A", "#2563EB", "#10B981", "#F8FAFC"]

class Finding(BaseModel):
    id: str
    category: Literal["terminology", "fact_claim", "tone", "visual", "accessibility", "cross_modal"]
    severity: Literal["high", "medium", "low"]
    confidence: int = Field(..., ge=0, le=100, description="0-100 confidence score")
    original_text: str
    explanation: str
    source_rule: str
    suggested_fix: str

class ReviewResponse(BaseModel):
    submission_id: str
    version_hash: str
    status: Literal["Draft", "AI Reviewed", "Changes Requested", "Approved", "Rejected"]
    compliance_score: int
    extracted_text: str
    findings: List[Finding]
    timestamp: str

class DismissalRequest(BaseModel):
    submission_id: str
    finding_id: str
    reason: str
    user_role: Literal["Creator", "Reviewer", "Admin"]

class ApprovalAction(BaseModel):
    submission_id: str
    version_hash: str
    reviewer_name: str
    action: Literal["Changes Requested", "Approved", "Rejected"]
    comment: Optional[str] = None