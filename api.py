"""HTTP API for extracting device-label details from an uploaded image."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from ocr import extract_device_attributes

app = FastAPI(title="Device Label OCR API", version="1.0.0")

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_FILES_PER_REQUEST = 10
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
DEBUG_UPLOAD_DIR = os.getenv("DEBUG_UPLOAD_DIR")


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight endpoint for deployment health checks."""
    return {"status": "ok"}


@app.post("/extract")
async def extract(
    files: Annotated[list[UploadFile] | None, File()] = None,
    file: Annotated[UploadFile | None, File()] = None,
    image: Annotated[UploadFile | None, File()] = None,
) -> dict:
    """Extract attributes from uploads named ``files``, ``file``, or ``image``."""
    uploads = list(files or [])
    uploads.extend(upload for upload in (file, image) if upload is not None)
    if not uploads:
        raise HTTPException(
            status_code=422,
            detail="Upload an image using the multipart field name files, file, or image.",
        )
    if len(uploads) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Upload no more than {MAX_FILES_PER_REQUEST} images per request.",
        )

    results = []
    for upload in uploads:
        results.append(await extract_one(upload))
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
    debug_image_id = None
    try:
        if DEBUG_UPLOAD_DIR:
            debug_directory = Path(DEBUG_UPLOAD_DIR)
            debug_directory.mkdir(parents=True, exist_ok=True)
            debug_image_id = f"{uuid4().hex}{suffix}"
            (debug_directory / debug_image_id).write_bytes(image_bytes)

        with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name
        result = {
            "filename": file.filename,
            "attributes": extract_device_attributes(temp_path),
        }
        if debug_image_id:
            result["debug_image_id"] = debug_image_id
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc
    finally:
        await file.close()
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
