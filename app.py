import os
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


def generate_medical_description(image_bytes: bytes, mime_type: str) -> str:
    if client is None:
        raise RuntimeError("Missing GOOGLE_API_KEY in .env file.")

    prompt = """
    You are a clinical imaging assistant.
    Generate a detailed, medically grounded description of the provided image.
    Include:
    1) observed structures and findings,
    2) possible abnormalities (if any),
    3) cautious interpretation notes,
    4) next-step recommendations.
    Avoid fabricated certainty. If uncertain, clearly state limitations.
    """

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
    )
    return (response.text or "").strip()


def is_medical_response(description: str) -> bool:
    if client is None:
        raise RuntimeError("Missing GOOGLE_API_KEY in .env file.")

    validation_prompt = f"""
    Decide if this description is genuinely medical/clinical content.
    Reply with exactly one word: Yes or No.

    Description:
    {description}
    """

    result = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=validation_prompt,
    )
    answer = (result.text or "").strip().lower()
    return answer.startswith("yes")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", page_context(request))


@app.post("/", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    if not file.filename:
        return templates.TemplateResponse(
            "index.html",
            page_context(request, error_message="Please choose an image before submitting."),
        )

    mime_type = (file.content_type or "").lower()
    if mime_type not in ALLOWED_MIME_TYPES:
        return templates.TemplateResponse(
            "index.html",
            page_context(
                request,
                error_message="Unsupported image type. Please upload JPG, JPEG, PNG, JFIF, or WEBP.",
                uploaded_filename=file.filename,
            ),
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            return templates.TemplateResponse(
                "index.html",
                page_context(request, error_message="The uploaded file is empty.", uploaded_filename=file.filename),
            )

        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            return templates.TemplateResponse(
                "index.html",
                page_context(
                    request,
                    error_message="File too large. Maximum allowed size is 8 MB.",
                    uploaded_filename=file.filename,
                ),
            )

        description = generate_medical_description(image_bytes=image_bytes, mime_type=mime_type)
        if not description:
            return templates.TemplateResponse(
                "index.html",
                page_context(
                    request,
                    error_message="No description was generated. Please try another image.",
                    uploaded_filename=file.filename,
                ),
            )

        if not is_medical_response(description):
            return templates.TemplateResponse(
                "index.html",
                page_context(
                    request,
                    error_message="The image does not appear to be medical. Please upload a valid medical image.",
                    uploaded_filename=file.filename,
                ),
            )

        return templates.TemplateResponse(
            "index.html",
            page_context(
                request,
                response_text=description,
                status_message="Analysis generated successfully.",
                uploaded_filename=file.filename,
            ),
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            page_context(
                request,
                error_message=f"Could not process image: {exc}",
                uploaded_filename=file.filename,
            ),
        )