import time
import re
import base64
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types

app = FastAPI(title="BrandGuard AI Gateway")

# Enable CORS for Vercel Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client (Reads GEMINI_API_KEY from environment)
client = genai.Client()

class ReviewRequest(BaseModel):
    text: str
    brand_rules: Optional[str] = "Tone: Empathetic & Professional. Never mention rival 'AcmeAI'. Never promise 100% guaranteed ROI."
    image_base64: Optional[str] = None

class ReviewResponse(BaseModel):
    risk_score: int
    status: str
    violations: List[str]
    sanitized_rewrite: str
    explanation: str

@app.post("/api/review/analyze")
async def analyze_submission(req: ReviewRequest):
    t0 = time.perf_counter()

    # Stage 1: Fast Regex PII Detection (<5ms)
    fast_redacted = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', req.text)
    fast_redacted = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', fast_redacted)
    stage1_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Stage 2: Gemini 2.5 Flash Structured Evaluation
    t1 = time.perf_counter()
    prompt_content = f"""
    Analyze the following submission against safety guidelines and custom brand rules.
    Brand Guidelines: {req.brand_rules}
    Input Text: {req.text}
    Pre-sanitized Text: {fast_redacted}

    If violations occur (PII, off-brand claims, competitor mentions, toxicity), assign a risk score between 0 and 100, list violations, and provide a compliant 'sanitized_rewrite'.
    """

    contents = [prompt_content]

    # Process image if uploaded
    if req.image_base64:
        try:
            image_data = req.image_base64.split(",")[-1]
            img_bytes = base64.b64decode(image_data)
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReviewResponse,
                temperature=0.1,
            ),
        )
        stage2_ms = round((time.perf_counter() - t1) * 1000, 2)
        
        result = ReviewResponse.model_validate_json(response.text).model_dump()
        result["telemetry"] = {
            "stage1_regex_ms": stage1_ms,
            "stage2_gemini_flash_ms": stage2_ms
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")