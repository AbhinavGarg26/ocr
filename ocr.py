"""
Reads a photo of a device label (e.g. a Motorola retail box label) and
extracts structured attributes via OCR.

Requirements (install once):
    sudo apt-get install tesseract-ocr
    pip install pytesseract opencv-python-headless numpy pillow

Usage:
    python3 label_ocr.py /path/to/label_photo.jpg
"""

import re
import sys
import cv2
import numpy as np
import pytesseract

try:
    import zxingcpp
except ImportError:  # Allow text-only OCR when the optional decoder is absent.
    zxingcpp = None

# Each field's regex. Every pattern returns None if not found, rather than
# guessing -- so you know which fields need a manual check.
PATTERNS = {
    # Motorola's current labels commonly use codes such as ``XT2625-5``.
    # Keep the code independent from the model name: OCR can read one even
    # when the other is blurred or partially obscured.
    "model_code": r"(?<![A-Z0-9])(XT\s*\d{3,5}\s*-\s*\d{1,3})\b",
    "model_name": r"\b((?:moto|motorola)\s+[a-z]\s*\d{1,3}(?:\s*(?:5g|4g))?)\b",
    "mo_number": r"MO#:?\s*(\d{6,})",
    "ram_storage": r"\b(\d{1,3})\s*\+\s*(\d{1,4})\s*G\s*B\b",
    # Some OCR layouts split the printed word "COLOR" into "CO LO R".
    "color": r"\bC\s*O\s*L\s*O\s*R\s*:?\s*([A-Za-z][A-Za-z .'-]*\([A-Za-z .'-]+\))",
    # Sparse OCR may insert spaces between digits, so accept whitespace within
    # a 15-digit IMEI and normalise it after matching.
    "imei1": r"IMEI\s*1:?\s*((?:\d\s*){15})",
    "imei2": r"IMEI\s*2:?\s*((?:\d\s*){15})",
    "part_number": r"(?:P/?N|SKU):?\s*([A-Z0-9-]{6,})",
    "bis_registration": r"(R-\d{6,8})",
    "sar_head": r"SAR:?\s*([\d.]+)\s*W/?kg@1g\s*\(?HEAD\)?",
    "sar_body": r"([\d.]+)\s*W/?kg@1g\s*\(?BODY\)?",
    "battery_epr": r"Battery EPR No\.?:?\s*(\d{10,})",
    "made_in": r"MADE IN (\w+)",
    "manufactured_for": r"(MOTOROLA MOBILITY[^\n]*)",
    # Laptop labels vary widely, so use their standard field labels rather
    # than a vendor-specific model-number format.
    "display": r"(?:DISPLAY|SCREEN|LCD)\s*:?[ \t]*([^\n]+)",
    "chip": r"(?:CHIP|PROCESSOR|CPU|SOC)\s*:?[ \t]*([^\n]+)",
    "serial_number": r"(?:SERIAL(?:\s+(?:NUMBER|NO\.?))?|S/?N)\s*:?\s*([A-Z0-9-]{5,})",
}

# Use a small override table only when the printed marketing name is too
# degraded for OCR.  The normal regex above remains the generic path for
# labels where the name is legible; add future Motorola XT-code mappings here.
MOTOROLA_MODEL_NAMES = {
    "XT2625-5": "moto g37",
}

OUTPUT_FIELDS = (
    "model_code", "model_name", "color", "imei1", "imei2", "ram",
    "storage", "part_number", "display", "chip", "serial_number",
)


def preprocess(image_path: str) -> np.ndarray:
    """Grayscale, denoise, contrast-boost and upscale for better OCR accuracy."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE (adaptive contrast) handles uneven lighting/glare better than a
    # flat autocontrast would.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(contrasted, h=10)

    # Upscale -- Tesseract does noticeably better on larger text.
    upscaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Adaptive threshold to binarize, which helps with the glossy label glare.
    thresh = cv2.adaptiveThreshold(
        upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return thresh


def ocr_text(image_path: str) -> str:
    """Return text from complementary OCR passes.

    A thresholded dense-text pass is good for barcodes and small print, while
    a sparse pass on the original photo better preserves larger label fields
    such as colour and memory configuration.  Combining them makes the parser
    work for Motorola retail labels without tying it to one phone model.
    """
    processed = preprocess(image_path)
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    dense_text = pytesseract.image_to_string(processed, config="--psm 6")
    sparse_text = pytesseract.image_to_string(original, config="--psm 3")
    label_text = pytesseract.image_to_string(original, config="--psm 11")
    return f"{dense_text}\n{sparse_text}\n{label_text}"


def extract_attributes(text: str) -> dict:
    attributes = {}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        attributes[key] = match.group(1).strip() if match else None

    # ram_storage has two capture groups; expand into two fields.
    ram_match = re.search(PATTERNS["ram_storage"], text, re.IGNORECASE)
    attributes["ram"] = f"{ram_match.group(1)}GB" if ram_match else None
    attributes["storage"] = f"{ram_match.group(2)}GB" if ram_match else None
    del attributes["ram_storage"]

    # Normalise harmless OCR whitespace in Motorola part/model codes.
    if attributes["model_code"]:
        attributes["model_code"] = re.sub(r"\s+", "", attributes["model_code"])
        if not attributes["model_name"]:
            attributes["model_name"] = MOTOROLA_MODEL_NAMES.get(attributes["model_code"])

    for field in ("imei1", "imei2"):
        if attributes[field]:
            attributes[field] = re.sub(r"\D", "", attributes[field])

    # Generic retail-label fallbacks.  These cover labels such as Google Pixel
    # boxes, where the model is printed as "Model: GUJ0N" rather than an XT
    # code and the configuration is presented in a compact pipe-separated row.
    if not attributes["model_code"]:
        match = re.search(r"\bMODEL\s*:?\s*([A-Z0-9-]{4,})\b", text, re.IGNORECASE)
        attributes["model_code"] = match.group(1).upper() if match else None

    if not attributes["model_name"]:
        match = re.search(r"\b(GOOGLE\s+PIXEL\s+\d{1,2}(?:A)?(?:\s+(?:PRO|XL|FOLD))?)\b", text, re.IGNORECASE)
        if match:
            attributes["model_name"] = re.sub(r"(?<=\d)A\b", "a", match.group(1).title())

    if not attributes["storage"]:
        match = re.search(r"\b(\d{2,4})\s*(GB|TB)\*?\b", text, re.IGNORECASE)
        attributes["storage"] = f"{match.group(1)}{match.group(2).upper()}" if match else None

    if not attributes["color"]:
        # Example: "5G Sub-6 | 6.3\" | 256 GB | Pistachio"
        match = re.search(r"\|\s*\d{2,4}\s*(?:GB|TB)\*{0,2}\s*\|\s*([A-Za-z][A-Za-z -]+)", text, re.IGNORECASE)
        attributes["color"] = match.group(1).strip() if match else None

    if not attributes["display"]:
        match = re.search(r"\b(\d{1,2}(?:\.\d+)?)\s*(?:\"|INCH(?:ES)?)", text, re.IGNORECASE)
        attributes["display"] = f"{match.group(1)}-inch" if match else None

    return attributes


def is_valid_imei(imei: str) -> bool:
    """Luhn checksum validation -- lets you flag OCR misreads automatically."""
    if not imei or not imei.isdigit() or len(imei) != 15:
        return False
    digits = [int(d) for d in imei]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def imeis_from_barcodes(image_path: str) -> list[str]:
    """Decode valid IMEIs from 1D/2D barcodes on a device label.

    ZXing recognizes Code 128, EAN/UPC and QR codes.  Only a valid 15-digit
    Luhn number is returned, so EAN product barcodes and support URLs cannot
    be mistaken for an IMEI.
    """
    if zxingcpp is None:
        return []

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    imeis = []
    for barcode in zxingcpp.read_barcodes(image):
        for candidate in re.findall(r"(?<!\d)\d{15}(?!\d)", barcode.text):
            if is_valid_imei(candidate) and candidate not in imeis:
                imeis.append(candidate)
    return imeis


def fill_missing_imeis_from_barcodes(attributes: dict, image_path: str) -> set[str]:
    """Fill absent or checksum-invalid IMEI fields using decoded barcodes."""
    barcode_imeis = imeis_from_barcodes(image_path)
    known_imeis = {
        attributes[field]
        for field in ("imei1", "imei2")
        if is_valid_imei(attributes.get(field))
    }
    recovered_fields = set()

    for field in ("imei1", "imei2"):
        if is_valid_imei(attributes.get(field)):
            continue
        replacement = next((imei for imei in barcode_imeis if imei not in known_imeis), None)
        if replacement:
            attributes[field] = replacement
            known_imeis.add(replacement)
            recovered_fields.add(field)
    return recovered_fields


def extract_device_attributes(image_path: str) -> dict:
    """Extract the public API fields from one device-label image."""
    attributes = extract_attributes(ocr_text(image_path))
    fill_missing_imeis_from_barcodes(attributes, image_path)
    return {field: attributes.get(field) for field in OUTPUT_FIELDS}


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 label_ocr.py /path/to/image.jpg")

    image_path = sys.argv[1]
    attributes = extract_device_attributes(image_path)

    print("Extracted attributes:")
    for key in OUTPUT_FIELDS:
        value = attributes[key]
        # Keep the CLI focused on the requested fields; absent optional laptop
        # fields are omitted rather than producing noisy "not found" lines.
        if value is None:
            continue
        note = ""
        if key in ("imei1", "imei2") and value:
            note = "  [checksum OK]" if is_valid_imei(value) else "  [checksum FAILED -- likely OCR misread]"
        print(f"  {key:20s} {value}{note}")


if __name__ == "__main__":
    main()
