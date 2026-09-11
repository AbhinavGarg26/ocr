"""HTTP API for extracting device-label details from an uploaded image."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile

from ocr import extract_device_attributes

app = FastAPI(title="Device Label OCR API", version="1.0.0")

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_FILES_PER_REQUEST = 10
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight endpoint for deployment health checks."""
    return {"status": "ok"}


@app.post("/extract")
async def extract(files: list[UploadFile] = File(...)) -> dict:
    """Extract key-value device details from one or more uploaded images."""
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Upload no more than {MAX_FILES_PER_REQUEST} images per request.",
        )

    results = []
    for file in files:
        results.append(await extract_one(file))
    return {"results": results}


async def extract_one(file: UploadFile) -> dict:
    """Extract attributes for one upload and remove its temporary file."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, JPEG, PNG, WEBP, BMP, or TIFF image.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 20 MB or smaller.")

    temp_path = None
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name
        return {
            "filename": file.filename,
            "attributes": extract_device_attributes(temp_path),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc
    finally:
        await file.close()
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
