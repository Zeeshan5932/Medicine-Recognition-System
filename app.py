import os
import re
import time
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="Medicine Recognition System")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024
PRIMARY_MODEL = "gemini-3-flash-preview"
FALLBACK_MODEL = "gemini-2.5-flash"
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
        raise RuntimeError(
            "The AI service is busy right now. Please try again in a minute."
        ) from last_error

    raise RuntimeError("The AI service is busy right now. Please try again in a minute.")


def generate_medical_description(image_bytes: bytes, mime_type: str) -> str:
    prompt = """
    You are a clinical imaging assistant.
    Generate a detailed, medically grounded description of the provided image.
    Include:
    1. Observed structures and findings
    2. Possible abnormalities, if any
    3. Cautious interpretation notes
    4. Next-step recommendations
    5. A short final summary
    Write in plain text only.
    Do not use markdown symbols such as #, *, or **.
    Use clear section titles and short bullet points with hyphens only if needed.
    Avoid fabricated certainty. If uncertain, clearly state limitations.
    """

    return generate_with_retry(
        prompt=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        model_sequence=(PRIMARY_MODEL, FALLBACK_MODEL),
    )


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


@app.post("/", response_class=HTMLResponse)
@app.post("/index.html", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    if not file.filename:
        return render_index(request, error_message="Please choose an image before submitting.")

    mime_type = (file.content_type or "").lower()
    if mime_type not in ALLOWED_MIME_TYPES:
        return render_index(
            request,
            error_message="Unsupported image type. Please upload JPG, JPEG, PNG, JFIF, or WEBP.",
            uploaded_filename=file.filename,
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            return render_index(request, error_message="The uploaded file is empty.", uploaded_filename=file.filename)

        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            return render_index(
                request,
                error_message="File too large. Maximum allowed size is 8 MB.",
                uploaded_filename=file.filename,
            )

        description = generate_medical_description(image_bytes=image_bytes, mime_type=mime_type)
        if not description:
            return render_index(
                request,
                error_message="No description was generated. Please try another image.",
                uploaded_filename=file.filename,
            )

        if not is_medical_response(description):
            return render_index(
                request,
                error_message="The image does not appear to be medical. Please upload a valid medical image.",
                uploaded_filename=file.filename,
            )

        return render_index(
            request,
            response_text=description,
            status_message="Analysis generated successfully.",
            uploaded_filename=file.filename,
        )
    except Exception as exc:
        return render_index(request, error_message=f"Could not process image: {exc}", uploaded_filename=file.filename)