import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("med-recognition")

app = FastAPI(title="Medicine Recognition System")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024
PRIMARY_MODEL = "gemini-3-flash-preview"
FALLBACK_MODEL = "gemini-2.5-flash"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/jfif",
    "image/webp",
}

api_key = os.getenv("GOOGLE_API_KEY", "").strip()
client = genai.Client(api_key=api_key) if api_key else None


def page_context(request: Request, **extra: Any) -> Dict[str, Any]:
    base_context: Dict[str, Any] = {
        "request": request,
        "response_text": None,
        "analysis_result": None,
        "error_message": None,
        "status_message": None,
        "uploaded_filename": None,
    }
    base_context.update(extra)
    return base_context


def clean_analysis_text(raw_text: str) -> str:
    text = raw_text.strip()
    text = text.replace("**", "")
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[•*\-]\s+", "- ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "uploaded_image"


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        return {}

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retryable_signals = (
        "503",
        "unavailable",
        "429",
        "resource_exhausted",
        "deadline_exceeded",
        "internal",
    )
    return any(signal in message for signal in retryable_signals)


def humanize_ai_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "missing google_api_key" in message:
        return "Server configuration is incomplete. Please set GOOGLE_API_KEY in .env."
    if "429" in message or "quota" in message or "resource_exhausted" in message:
        return "API rate limit reached. Please wait a moment and try again."
    if "timeout" in message or "deadline_exceeded" in message:
        return "The AI request timed out. Please try with a clearer image."
    if "503" in message or "unavailable" in message:
        return "AI service is temporarily busy. Please retry in 30 to 60 seconds."
    return "Could not process the image right now. Please try again shortly."


def generate_with_retry(*, prompt: Any, model_sequence: tuple[str, ...]) -> str:
    if client is None:
        raise RuntimeError("Missing GOOGLE_API_KEY in .env file.")

    last_error: Exception | None = None

    for model_name in model_sequence:
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                return clean_analysis_text(response.text or "")
            except Exception as exc:
                last_error = exc
                if not is_retryable_error(exc) or attempt == 2:
                    break
                time.sleep(1.5 * (attempt + 1))

    if last_error is not None:
        raise RuntimeError(humanize_ai_error(last_error)) from last_error

    raise RuntimeError("Could not process the image right now. Please try again shortly.")


def classify_image_type(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    prompt = """
    Classify the uploaded image and respond only as JSON with keys:
    image_type: one of [X-ray, MRI, CT, skin image, medicine packaging, dental image, non-medical, unknown]
    is_medical: true or false
    confidence_score: number from 0 to 1
    """

    raw = generate_with_retry(
        prompt=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
        model_sequence=(PRIMARY_MODEL, FALLBACK_MODEL),
    )
    data = extract_json(raw)

    image_type = str(data.get("image_type", "unknown")).strip() or "unknown"
    is_medical = bool(data.get("is_medical", image_type.lower() != "non-medical"))
    confidence = float(data.get("confidence_score", 0.5))
    confidence = max(0.0, min(confidence, 1.0))

    return {
        "image_type": image_type,
        "is_medical": is_medical,
        "confidence_score": confidence,
    }


def build_analysis_prompt(image_type: str) -> str:
    return f"""
    You are a clinical imaging assistant.
    Image category detected: {image_type}

    Provide analysis in JSON only with keys:
    short_summary: short paragraph, plain text
    detailed_description: clear, structured explanation in plain text
    confidence_score: number from 0 to 1
    warning: short medical disclaimer

    Keep tone professional and cautious. No markdown symbols.
    """


def generate_medical_description(image_bytes: bytes, mime_type: str) -> str:
    return generate_with_retry(
        prompt=[build_analysis_prompt("unknown"), types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
        model_sequence=(PRIMARY_MODEL, FALLBACK_MODEL),
    )


def generate_structured_analysis(image_bytes: bytes, mime_type: str, image_type: str) -> Dict[str, Any]:
    raw = generate_with_retry(
        prompt=[build_analysis_prompt(image_type), types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
        model_sequence=(PRIMARY_MODEL, FALLBACK_MODEL),
    )
    data = extract_json(raw)

    if not data:
        cleaned = clean_analysis_text(raw)
        return {
            "short_summary": "AI generated an analysis for the uploaded image.",
            "detailed_description": cleaned,
            "confidence_score": 0.6,
            "warning": "AI output is informational only and not a confirmed diagnosis.",
        }

    return {
        "short_summary": clean_analysis_text(str(data.get("short_summary", "")).strip()) or "No summary provided.",
        "detailed_description": clean_analysis_text(str(data.get("detailed_description", "")).strip())
        or "No detailed description provided.",
        "confidence_score": max(0.0, min(float(data.get("confidence_score", 0.6)), 1.0)),
        "warning": clean_analysis_text(str(data.get("warning", "")).strip())
        or "AI output is informational only and not a confirmed diagnosis.",
    }


def is_medical_response(description: str) -> bool:
    validation_prompt = f"""
    Decide if this description is genuinely medical/clinical content.
    Reply with exactly one word: Yes or No.

    Description:
    {description}
    """

    answer = generate_with_retry(prompt=validation_prompt, model_sequence=(PRIMARY_MODEL, FALLBACK_MODEL)).lower()
    return answer.startswith("yes")


def render_index(request: Request, **extra: Any) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", page_context(request, **extra))


@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", page_context(request))


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "medicine-recognition", "model": PRIMARY_MODEL}


@app.post("/", response_class=HTMLResponse)
@app.post("/index.html", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    started_at = time.perf_counter()

    if not file.filename or not file.filename.strip():
        return render_index(request, error_message="Please choose an image before submitting.")

    safe_name = sanitize_filename(file.filename)
    extension = get_extension(safe_name)
    mime_type = (file.content_type or "").lower()

    if extension not in ALLOWED_EXTENSIONS:
        return render_index(
            request,
            error_message="Invalid file extension. Please upload JPG, JPEG, PNG, JFIF, or WEBP.",
            uploaded_filename=safe_name,
        )

    if mime_type and mime_type not in ALLOWED_MIME_TYPES:
        return render_index(
            request,
            error_message="Unsupported image type. Please upload a valid image file.",
            uploaded_filename=safe_name,
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            return render_index(request, error_message="The uploaded file is empty.", uploaded_filename=safe_name)

        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            return render_index(
                request,
                error_message="File too large. Maximum allowed size is 8 MB.",
                uploaded_filename=safe_name,
            )

        effective_mime = mime_type or "image/jpeg"
        classification = classify_image_type(image_bytes=image_bytes, mime_type=effective_mime)
        structured = generate_structured_analysis(
            image_bytes=image_bytes,
            mime_type=effective_mime,
            image_type=classification["image_type"],
        )

        ai_medical = is_medical_response(structured["detailed_description"])
        is_medical = bool(classification["is_medical"] and ai_medical)
        confidence = round((classification["confidence_score"] + structured["confidence_score"]) / 2, 2)

        if not is_medical:
            return render_index(
                request,
                error_message="The image does not appear to be medical. Please upload a valid medical image.",
                uploaded_filename=safe_name,
                analysis_result={
                    "image_type": classification["image_type"],
                    "is_medical": False,
                    "short_summary": "Image rejected as non-medical.",
                    "detailed_description": "The uploaded content could not be confirmed as medical context.",
                    "confidence_score": confidence,
                    "warning": "Use a clear medical image for best results.",
                },
            )

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "analysis_success filename=%s image_type=%s confidence=%.2f duration_ms=%s",
            safe_name,
            classification["image_type"],
            confidence,
            elapsed_ms,
        )

        result_payload = {
            "image_type": classification["image_type"],
            "is_medical": True,
            "short_summary": structured["short_summary"],
            "detailed_description": structured["detailed_description"],
            "confidence_score": confidence,
            "warning": structured["warning"],
        }

        return render_index(
            request,
            response_text=structured["detailed_description"],
            analysis_result=result_payload,
            status_message="Analysis generated successfully.",
            uploaded_filename=safe_name,
        )
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.error("analysis_failed filename=%s duration_ms=%s error=%s", safe_name, elapsed_ms, exc)
        return render_index(
            request,
            error_message=humanize_ai_error(exc),
            uploaded_filename=safe_name,
        )