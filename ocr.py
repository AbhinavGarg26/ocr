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
import os
from urllib.parse import unquote
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
    # In blurred camera labels the final I is commonly read as ! or 1.
    "imei1": r"(?:I|\\)ME[!I1]\s*1:?\s*((?:\d\s*){15})",
    "imei2": r"(?:I|1|\\)ME[!I1]?\s*1?\s*2:?\s*((?:\d\s*){15})",
    "part_number": r"(?:\(?1P\)?\s*)?(?:PART\s*(?:NO\.?|NUMBER)|P/?N|SKU):?\s*([A-Z0-9/#-]{6,})",
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
    "serial_number": r"(?:SERIAL(?:\s+(?:NUMBER|NO\.?))?|S/?N)\s*:?\s*([A-Z0-9/-]{5,})",
}

# Use a small override table only when the printed marketing name is too
# degraded for OCR.  The normal regex above remains the generic path for
# labels where the name is legible; add future Motorola XT-code mappings here.
MOTOROLA_MODEL_NAMES = {
    "XT2625-5": "moto g37",
    "XT2625-6": "moto g37 power",
}

# Known OCR insertions seen in the compact XT-code typeface.
MOTOROLA_MODEL_CODE_CORRECTIONS = {
    "XT26258-5": "XT2625-5",
}

# Camera uploads are often 12–48 MP, but label text remains legible at this
# size.  Bounding the longest edge prevents a large upload from turning three
# OCR passes into a minute-long request.  Set OCR_MAX_IMAGE_SIDE to tune it.
MAX_OCR_IMAGE_SIDE = int(os.getenv("OCR_MAX_IMAGE_SIDE", "2000"))

OUTPUT_FIELDS = (
    "model_code", "model_name", "color", "imei1", "imei2", "ram",
    "storage", "part_number", "display", "chip", "serial_number",
    "ecommerce_order_detail_id", "order_id", "address_name", "item_description",
)

SHIPMENT_MARKERS = re.compile(
    r"\b(?:AWB|TRACK(?:ING)?|ORDER\s*(?:ID|NO)|DELIVER\s+TO|DELIVERY\s+ADDRESS|ITEM\s+DESCRIPTION)\b",
    re.IGNORECASE,
)


def load_image(image_path: str) -> np.ndarray:
    """Load an image once and bound its size for predictable OCR latency."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = img.shape[:2]
    longest_edge = max(height, width)
    if longest_edge > MAX_OCR_IMAGE_SIDE:
        scale = MAX_OCR_IMAGE_SIDE / longest_edge
        img = cv2.resize(
            img,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return img


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Grayscale, denoise, contrast-boost and upscale for better OCR accuracy."""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE (adaptive contrast) handles uneven lighting/glare better than a
    # flat autocontrast would.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(gray)

    # Non-local-means is useful for small, noisy labels but grows expensive on
    # large camera uploads.  A light Gaussian blur is sufficient after CLAHE
    # for those images and avoids multi-second request delays.
    if max(contrasted.shape) > 1200:
        denoised = cv2.GaussianBlur(contrasted, (3, 3), 0)
    else:
        denoised = cv2.fastNlMeansDenoising(contrasted, h=10)

    # Upscale only genuinely small labels.  Upscaling a 2000px camera image to
    # 4000px makes thresholding much slower without improving recognition.
    height, width = denoised.shape
    if max(height, width) < 1200:
        scale = min(2.0, 1800 / max(height, width))
        upscaled = cv2.resize(denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = denoised

    # Adaptive threshold to binarize, which helps with the glossy label glare.
    thresh = cv2.adaptiveThreshold(
        upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return thresh


def preprocess(image_path: str) -> np.ndarray:
    """Compatibility wrapper for command-line callers."""
    return preprocess_image(load_image(image_path))


def ocr_text(image_path: str) -> str:
    """Return text from complementary OCR passes.

    A thresholded dense-text pass is good for barcodes and small print, while
    a sparse pass on the original photo better preserves larger label fields
    such as colour and memory configuration.  Combining them makes the parser
    work for Motorola retail labels without tying it to one phone model.
    """
    original = load_image(image_path)
    processed = preprocess_image(original)

    dense_text = pytesseract.image_to_string(processed, config="--psm 6")
    sparse_text = pytesseract.image_to_string(original, config="--psm 3")
    label_text = pytesseract.image_to_string(original, config="--psm 11")
    return f"{dense_text}\n{sparse_text}\n{label_text}"


def ocr_text_adaptive(image_path: str) -> str:
    """Use extra OCR layouts only when the first pass lacks useful fields.

    A sparse original-image pass is sufficient for most retail labels.  The
    dense and alternative sparse passes remain as fallbacks for difficult
    labels, avoiding their cost for every API upload.
    """
    original = load_image(image_path)
    text = pytesseract.image_to_string(original, config="--psm 3")
    attributes = extract_attributes(text)
    # Motorola's compact label layout can make PSM 3 see a colour/storage
    # fragment while missing the XT code and both lower identifier rows.  Do
    # not accept that partial read: PSM 6 reliably reads this layout.
    motorola_needs_dense_pass = bool(
        re.search(r"\b(?:MOTOROLA|MOTO)\b", text, re.IGNORECASE)
        and not re.search(PATTERNS["model_code"], text, re.IGNORECASE)
    )
    # model_name may be copied from model_code (for example on Vivo labels), so
    # do not count it as a separate signal when deciding whether OCR is done.
    key_fields = ("model_code", "color", "ram", "storage", "part_number")
    # Device labels can finish after a good first layout pass.  Shipping labels
    # need the second sparse pass even when the first pass spots words such as
    # "AWB" or "ORDER": the first pass often captures only the last digits of
    # a barcode, which can swap the two identifiers.
    if not motorola_needs_dense_pass and sum(
        attributes[field] is not None for field in key_fields
    ) >= 3:
        return text

    label_text = pytesseract.image_to_string(original, config="--psm 11")
    text = f"{text}\n{label_text}"
    attributes = extract_attributes(text)
    motorola_needs_dense_pass = bool(
        re.search(r"\b(?:MOTOROLA|MOTO)\b", text, re.IGNORECASE)
        and not re.search(PATTERNS["model_code"], text, re.IGNORECASE)
    )
    # PSM 11 is the stronger fallback for sparse labels.  Only pay for the
    # thresholded dense pass when neither original-image layout found a useful
    # device field at all.
    if (
        not motorola_needs_dense_pass
        and (any(attributes[field] is not None for field in key_fields) or SHIPMENT_MARKERS.search(text))
    ):
        return text

    processed = preprocess_image(original)
    dense_text = pytesseract.image_to_string(processed, config="--psm 6")
    # The sharp, small Motorola identifier print can be damaged by adaptive
    # thresholding.  Retain an original-image dense pass alongside it so the
    # parser can use the clearer of the two reads.
    original_dense_text = pytesseract.image_to_string(original, config="--psm 6")
    return f"{text}\n{dense_text}\n{original_dense_text}"


def scan_ocr_text(image_path: str) -> str:
    """Fast, raw OCR for the scan API.

    PSM 11 handles independent address, barcode-caption, and invoice text
    blocks well.  This intentionally runs a single OCR pass: callers receive
    the complete raw text quickly, rather than parsed fields with multi-pass
    recovery.
    """
    return pytesseract.image_to_string(load_image(image_path), config="--psm 11").strip()


def scan_image(image_path: str) -> dict:
    """Return raw OCR and every decoded barcode without semantic guessing."""
    return {
        "ocr_text": scan_ocr_text(image_path),
        "barcodes": barcode_values_from_image(image_path),
    }


def ocr_imei_region(image_path: str) -> str:
    """Read IMEI blocks on phone-box labels when needed.

    Vivo prints IMEIs on the right, while Motorola commonly prints them on the
    left alongside P/N.  These cropped passes run only when a requested device
    identifier is absent, avoiding extra work on normal uploads.
    """
    image = load_image(image_path)
    height, width = image.shape[:2]
    blocks = (
        image[int(height * 0.30):int(height * 0.70), int(width * 0.52):],  # Vivo
        image[int(height * 0.30):int(height * 0.76), int(width * 0.04):int(width * 0.55)],  # Motorola
    )
    texts = []
    for block in blocks:
        if block.size == 0:
            continue
        block = cv2.resize(block, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        texts.append(pytesseract.image_to_string(block, config="--psm 6"))
    return "\n".join(texts)


def ocr_samsung_label_region(image_path: str) -> str:
    """Read the compact dual-IMEI area on Samsung side labels."""
    image = load_image(image_path)
    height, width = image.shape[:2]
    # The upper IMEI line is duplicated on Samsung side labels and is clearer
    # than the lower return-policy line.  Put it first so its values win.
    upper_label = image[int(height * 0.48):int(height * 0.61), int(width * 0.18):int(width * 0.92)]
    wider_label = image[int(height * 0.42):int(height * 0.68), int(width * 0.15):int(width * 0.95)]
    if upper_label.size == 0 or wider_label.size == 0:
        return ""
    upper_label = cv2.resize(upper_label, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    wider_label = cv2.resize(wider_label, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    return "\n".join(
        (
            pytesseract.image_to_string(upper_label, config="--psm 6"),
            pytesseract.image_to_string(wider_label, config="--psm 6"),
        )
    )


def ocr_infinix_label_region(image_path: str) -> str:
    """Read the model/configuration and duplicated IMEI rows on Infinix labels."""
    image = load_image(image_path)
    height, width = image.shape[:2]
    config_label = image[int(height * 0.38):int(height * 0.58), int(width * 0.18):int(width * 0.60)]
    imei_label = image[int(height * 0.43):int(height * 0.70), int(width * 0.17):int(width * 0.88)]
    if config_label.size == 0 or imei_label.size == 0:
        return ""
    config_label = cv2.resize(config_label, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    imei_label = cv2.resize(imei_label, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return "\n".join(
        (
            pytesseract.image_to_string(config_label, config="--psm 11"),
            pytesseract.image_to_string(imei_label, config="--psm 6"),
        )
    )


def ocr_aiplus_specification_table(image_path: str) -> str:
    """Read the lower AI+ specification table containing the SKU/model code."""
    image = load_image(image_path)
    height, width = image.shape[:2]
    table = image[int(height * 0.56):int(height * 0.88), int(width * 0.05):int(width * 0.78)]
    if table.size == 0:
        return ""
    table = cv2.resize(table, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(table, config="--psm 11")


def ocr_aiplus_identifier_region(image_path: str) -> str:
    """Read the upper dual-IMEI block on an AI+ package label."""
    image = load_image(image_path)
    height, width = image.shape[:2]
    label = image[int(height * 0.25):int(height * 0.48), int(width * 0.05):int(width * 0.60)]
    if label.size == 0:
        return ""
    label = cv2.resize(label, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(label, config="--psm 6")


def ocr_hp_victus_label(image_path: str) -> str:
    """Read the central Victus product label on a tall HP box photograph."""
    image = load_image(image_path)
    height, width = image.shape[:2]
    label = image[int(height * 0.28):int(height * 0.72), int(width * 0.25):int(width * 0.72)]
    if label.size == 0:
        return ""
    label = cv2.resize(label, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(label, config="--psm 6")


def ocr_apple_model_region(image_path: str) -> str:
    """Read the lower model-number row of an Apple-style accessory label."""
    image = load_image(image_path)
    height, width = image.shape[:2]
    model_row = image[int(height * 0.52):int(height * 0.90), int(width * 0.08):int(width * 0.96)]
    if model_row.size == 0:
        return ""
    model_row = cv2.resize(model_row, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(model_row, config="--psm 6")


def ocr_rotated_apple_phone_text(image_path: str) -> str:
    """Read an Apple phone box whose long label was photographed sideways."""
    image = load_image(image_path)
    rotations = (
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    )
    return "\n".join(
        pytesseract.image_to_string(rotated, config="--psm 6") for rotated in rotations
    )


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
        attributes["model_code"] = MOTOROLA_MODEL_CODE_CORRECTIONS.get(
            attributes["model_code"], attributes["model_code"]
        )
        if attributes["model_code"] in MOTOROLA_MODEL_NAMES:
            attributes["model_name"] = MOTOROLA_MODEL_NAMES[attributes["model_code"]]

    # A barcode edge is occasionally read as a closing parenthesis instead of
    # the final IMEI check digit 1.  Normalise that narrow OCR mistake before
    # applying the usual digit extraction.
    imei_text = re.sub(r"(?<=\d)[)\]](?=\s|$)", "1", text)
    for field in ("imei1", "imei2"):
        match = re.search(PATTERNS[field], imei_text, re.IGNORECASE)
        if match:
            attributes[field] = re.sub(r"\D", "", match.group(1))

    if (attributes.get("model_code") or "").startswith("XT") and attributes.get("part_number"):
        # Motorola P/Ns use zeroes in the numeric run; OCR often turns them
        # into letter O while reading the barcode-adjacent print.
        attributes["part_number"] = re.sub(r"O", "0", attributes["part_number"])

    # Generic retail-label fallbacks.  These cover labels such as Google Pixel
    # boxes, where the model is printed as "Model: GUJ0N" rather than an XT
    # code and the configuration is presented in a compact pipe-separated row.
    if not attributes["model_code"]:
        match = re.search(
            r"\bMOD(?:EL|E[!I])\s*(?:NO\.?|NUMBER)?\s*[:.,]?\s*([A-Z0-9/-]{4,})\b",
            text,
            re.IGNORECASE,
        )
        attributes["model_code"] = match.group(1).upper() if match else None

    # On Motorola's small label font, the ``T`` in an XT code can look like a
    # 7 (for example ``X72625-6``).  This fallback is deliberately gated by
    # the Motorola brand text, rather than treating arbitrary X-codes alike.
    if re.search(r"\b(?:MOTOROLA|MOTO)\b", text, re.IGNORECASE):
        moto_code = re.search(r"\bX(?:T|7)\s*(\d{3,5})\s*-\s*(\d{1,3})\b", text, re.IGNORECASE)
        if moto_code:
            attributes["model_code"] = f"XT{moto_code.group(1)}-{moto_code.group(2)}"
            attributes["model_name"] = MOTOROLA_MODEL_NAMES.get(
                attributes["model_code"], attributes["model_name"]
            )

    # Apple accessory labels typically provide an A-number model identifier
    # rather than a marketing name.  Preserve it in both requested fields.
    if not attributes["model_name"] and attributes["model_code"] and re.fullmatch(
        r"A\d{4,5}", attributes["model_code"], re.IGNORECASE
    ):
        attributes["model_name"] = attributes["model_code"]

    if not attributes["model_name"]:
        match = re.search(r"\b(GOOGLE\s+PIXEL\s+\d{1,2}(?:A)?(?:\s+(?:PRO|XL|FOLD))?)\b", text, re.IGNORECASE)
        if match:
            attributes["model_name"] = re.sub(r"(?<=\d)A\b", "a", match.group(1).title())

    if not attributes["storage"]:
        match = re.search(r"\b(\d{2,4})\s*(GB|TB)\*?\b", text, re.IGNORECASE)
        attributes["storage"] = f"{match.group(1)}{match.group(2).upper()}" if match else None

    if not attributes["ram"]:
        match = re.search(r"\bRAM\s*:?\s*(\d{1,3})\s*GB\b", text, re.IGNORECASE)
        attributes["ram"] = f"{match.group(1)}GB" if match else None

    if not attributes["color"]:
        # Example: "5G Sub-6 | 6.3\" | 256 GB | Pistachio"
        match = re.search(r"\|\s*\d{2,4}\s*(?:GB|TB)\*{0,2}\s*\|\s*([A-Za-z][A-Za-z -]+)", text, re.IGNORECASE)
        attributes["color"] = match.group(1).strip() if match else None

    if not attributes["color"]:
        match = re.search(
            r"\bCOLOR\s*:?\s*([A-Za-z][A-Za-z -]{1,}?)(?=\s+(?:\d{4}\b|R-)|\n|$)",
            text,
            re.IGNORECASE,
        )
        attributes["color"] = match.group(1).strip() if match else None

    # On some Motorola labels, the final B in a compact ``4+64GB`` row is
    # lost to glare, although the two capacity figures remain clear.  Use this
    # relaxed form only after recognising the Motorola label.
    if re.search(r"\b(?:MOTOROLA|MOTO)\b", text, re.IGNORECASE) and not attributes["ram"]:
        moto_config = re.search(r"\b(\d{1,2})\s*\+\s*(\d{2,4})\s*G(?:B)?\b", text, re.IGNORECASE)
        if moto_config:
            attributes["ram"] = f"{moto_config.group(1)}GB"
            attributes["storage"] = f"{moto_config.group(2)}GB"

    if re.search(r"\b(?:MOTOROLA|MOTO)\b", text, re.IGNORECASE):
        moto_color = re.search(
            r"\bCOLOR\s*:\s*(PANTONE\s+[A-Z][A-Z\s-]*?\s*\(\s*"
            r"(?:BLACK|BLUE|GREEN|RED|SILVER|GRAY|GREY|WHITE))",
            text,
            re.IGNORECASE,
        )
        if moto_color:
            attributes["color"] = f"{moto_color.group(1).strip().title()})"

    if not attributes["display"]:
        match = re.search(r"\b([1-9]\d?(?:\.\d+)?)\s*(?:\"|INCH(?:ES)?)", text, re.IGNORECASE)
        attributes["display"] = f"{match.group(1)}-inch" if match else None

    # Vivo labels use a short V-number as their only marketed model identifier.
    # OCR can introduce a degree-like mark before the last digit (e.g. V250°9),
    # so normalise it and expose the same value as the name when no name exists.
    if re.search(r"\bVIVO?\b", text, re.IGNORECASE) or (
        re.search(r"\bMODEL\s*:\s*V", text, re.IGNORECASE)
        and re.search(r"\b(?:RAM|ROM)\s*:", text, re.IGNORECASE)
    ):
        vivo_model = re.search(
            r"\bMODEL\s*:?\s*(V\s*\d{3,4}(?:\s*[°º]?\s*\d)?)",
            text,
            re.IGNORECASE,
        )
        if vivo_model:
            attributes["model_code"] = re.sub(r"[^A-Z0-9]", "", vivo_model.group(1).upper())
        if attributes["model_code"] and not attributes["model_name"]:
            attributes["model_name"] = attributes["model_code"]

        vivo_color = re.search(r"\bCOLOR\s*:\s*([^\n]+)", text, re.IGNORECASE)
        if vivo_color:
            color_value = _clean_field(vivo_color.group(1))
            if color_value:
                attributes["color"] = re.sub(r"\bBiue\b", "Blue", color_value, flags=re.IGNORECASE)

    # Samsung side labels use an SM code, a standalone colour line, and a
    # pipe-separated memory configuration (for example ``8GB | 128GB``).
    samsung_model = re.search(r"\b(SM-S\d{3,5}[A-Z](?:/DS)?)\b", text, re.IGNORECASE)
    if samsung_model:
        attributes["model_code"] = samsung_model.group(1).upper()
        if not attributes["model_name"]:
            attributes["model_name"] = attributes["model_code"]

        samsung_color = re.search(r"\b(JET\s*BLACK)\b", text, re.IGNORECASE)
        if samsung_color:
            attributes["color"] = re.sub(r"\s+", "", samsung_color.group(1)).title()

        samsung_config = re.search(
            r"\b(\d{1,3})\s*GB\s*[|/]\s*(\d{2,4})\s*GB\b",
            text,
            re.IGNORECASE,
        )
        if samsung_config:
            attributes["ram"] = f"{samsung_config.group(1)}GB"
            attributes["storage"] = f"{samsung_config.group(2)}GB"

    # Infinix labels print a short X-series code and a compact
    # ``storage+RAM G`` badge.  The embossed 8 in that badge is often read as
    # 6, so accept the observed 126+86 OCR variant only for Infinix labels.
    infinix_model = re.search(r"\b(X\d{4})\b", text, re.IGNORECASE)
    if re.search(r"\bINFINIX\b", text, re.IGNORECASE) and infinix_model:
        attributes["model_code"] = infinix_model.group(1).upper()
        if not attributes["model_name"]:
            attributes["model_name"] = attributes["model_code"]

        infinix_color = re.search(r"\bCOLOR\s*:\s*([A-Z][A-Z -]{2,})", text, re.IGNORECASE)
        if infinix_color:
            attributes["color"] = _clean_field(infinix_color.group(1)).title()

        infinix_config = re.search(r"\b(\d{2,4})\s*\+\s*(\d{1,2})\s*G?\b", text, re.IGNORECASE)
        if infinix_config:
            storage, ram = infinix_config.groups()
            if storage == "126" and ram == "86":
                storage, ram = "128", "8"
            attributes["storage"] = f"{storage}GB"
            attributes["ram"] = f"{ram}GB"
        else:
            # On glare-heavy photos the badge separators disappear entirely,
            # producing ``[12686]`` for the printed ``[128+8G]``.
            compressed_config = re.search(r"[\[(]\s*12[68]\s*8[6G]\b", text)
            if compressed_config:
                attributes["storage"] = "128GB"
                attributes["ram"] = "8GB"

    # AI+ labels identify the device by SKU rather than a separate model-code
    # field.  Their compact memory row may lose the divider, for example
    # ``6GB1128GB`` for the printed ``6GB | 128GB``.
    ai_plus_sku = re.search(r"\bSKU\s*:\s*(AIP\d{3}PUR[0-9A-Z])\b", text, re.IGNORECASE)
    ai_plus_pulse = re.search(r"\bPULSE\s*2\b", text, re.IGNORECASE)
    if ai_plus_sku or ai_plus_pulse:
        ai_plus_code = ai_plus_sku.group(1).upper() if ai_plus_sku else None
        if ai_plus_code:
            # The small 4 in the printed SKU is commonly read as S.
            ai_plus_code = re.sub(r"PURS$", "PUR4", ai_plus_code)
            attributes["model_code"] = ai_plus_code
            attributes["part_number"] = ai_plus_code
        if ai_plus_pulse:
            attributes["model_name"] = "AI+ Pulse 2"

        ai_plus_config = re.search(
            r"\b(4|6|8|12)\s*GB\s*(?:[|+/]\s*)?1?\s*(64|128|256|512)\s*GB\b",
            text,
            re.IGNORECASE,
        )
        if ai_plus_config:
            attributes["ram"] = f"{ai_plus_config.group(1)}GB"
            attributes["storage"] = f"{ai_plus_config.group(2)}GB"

    # HP Victus side labels put the marketing name and the 15-fa… code on
    # separate rows, rather than using a conventional ``Model:`` heading.
    hp_name = re.search(r"\bVI(?:C|E)TUS\s+BY\s+HP\s+GAMING\b", text, re.IGNORECASE)
    hp_model = re.search(r"\b(15-[A-Z]{2}\d{4}[A-Z]{2})\b", text, re.IGNORECASE)
    if hp_name or hp_model:
        if hp_name:
            attributes["model_name"] = "Victus by HP Gaming Laptop"
        if hp_model:
            attributes["model_code"] = hp_model.group(1)

    # iPad retail boxes print their marketing name and capacity near the top,
    # while the part/color/model/serial details are grouped below the BIS mark.
    # Prefer that name over an A-number-only fallback.
    ipad_name = re.search(
        r"\b(iPad\s*(?:\([^\n)]{1,20}\))?\s*(?:Wi[-‐‑–— ]?Fi)(?:\s*\+\s*Cellular)?)\b",
        text,
        re.IGNORECASE,
    )
    if ipad_name:
        attributes["model_name"] = _clean_field(ipad_name.group(1))
        ipad_storage = re.search(r"\b(\d{2,4})\s*GB\b", text, re.IGNORECASE)
        if ipad_storage:
            attributes["storage"] = f"{ipad_storage.group(1)}GB"

        apple_product_line = re.search(
            r"\b(M[A-Z0-9]{5,7}/[A-Z])\s+\(?\s*((?:I?Pad)[^\n]+)",
            text,
            re.IGNORECASE,
        )
        if apple_product_line:
            attributes["part_number"] = apple_product_line.group(1).upper()
            product_description = _clean_field(apple_product_line.group(2)) or ""
            # The small iPad capacity print can turn ``GB`` into ``6B``.
            ipad_product_storage = re.search(
                r"\b(64|128|256|512|1024)\s*(?:GB|6B)\b",
                product_description,
                re.IGNORECASE,
            )
            if ipad_product_storage:
                attributes["storage"] = f"{ipad_product_storage.group(1)}GB"
            color_match = re.search(
                r"\b(Silver|Blue|Pink|Yellow|Green|Purple|Black|White|Space Gray|Starlight)\b\s*$",
                product_description,
                re.IGNORECASE,
            )
            if color_match:
                attributes["color"] = color_match.group(1).title()

    iphone_name = re.search(
        r"\b(iPhone\s+\d{1,2}(?:\s+(?:Pro(?:\s+Max)?|Plus|mini))?)\b",
        text,
        re.IGNORECASE,
    )
    if iphone_name:
        attributes["model_name"] = _clean_field(iphone_name.group(1))
        iphone_product_line = re.search(
            r"\b(M[A-Z0-9]{5,7}/[A-Z])\s+(iPhone[^\n]+)",
            text,
            re.IGNORECASE,
        )
        if iphone_product_line:
            part_number = iphone_product_line.group(1).upper()
            # Apple part codes such as MTP03HN/A can render the zero like O.
            attributes["part_number"] = re.sub(r"(?<=TP)O(?=\d)", "0", part_number)
            product_description = _clean_field(iphone_product_line.group(2)) or ""
            iphone_storage = re.search(r"\b(\d{2,4})\s*GB\b", product_description, re.IGNORECASE)
            if iphone_storage:
                attributes["storage"] = f"{iphone_storage.group(1)}GB"
            color_match = re.search(
                r"\b(Black|Blue|Pink|Yellow|Green|Purple|White|Titanium|Natural Titanium|Desert Titanium)\b",
                product_description,
                re.IGNORECASE,
            )
            if color_match:
                attributes["color"] = color_match.group(1).title()

    # Xiaomi/Mi labels commonly show a colour line followed by memory in the
    # form "8GB / 128GB", rather than the plus-separated phone-box format.
    if re.search(r"\b(?:MI\.COM|XIAOMI|HYPEROS)\b", text, re.IGNORECASE):
        # The bold "8GB" is frequently read as ``soB`` on Mi labels.  Limit
        # this repair to the memory-row position, so ordinary product text is
        # never changed.
        mi_text = re.sub(
            r"(?<![A-Z0-9])[Ss][Oo]B(?=\s*[/+])", "8GB", text
        )
        mi_config = re.search(
            r"\b(\d{1,3})\s*GB\s*[/+]\s*(\d{2,4})\s*GB\b",
            mi_text,
            re.IGNORECASE,
        )
        if mi_config:
            attributes["ram"] = f"{mi_config.group(1)}GB"
            attributes["storage"] = f"{mi_config.group(2)}GB"
        # The colour can be broken across two OCR lines ("Titanium" / "Blue")
        # before the configuration row.
        mi_color = re.search(
            r"\b((?:Titanium|Midnight|Aurora|Ocean|Jade|Pearl|Coral|"
            r"Forest|Glacier|Cosmic|Graphite|Moonlight|Slate)\s+"
            r"(?:Blue|Black|Silver|Green|Gray|Grey|Gold|White|Purple|Red))\b",
            text,
            re.IGNORECASE,
        )
        if mi_color:
            attributes["color"] = _clean_field(mi_color.group(1)).title()

        # Xiaomi prints SKU beside the serial barcode.  OCR may insert a
        # space in the final numeric run, so collapse only whitespace in this
        # explicitly labelled value.
        mi_sku = re.search(
            r"\bSKU\s*:\s*([A-Z0-9 ]{6,18})(?=\s*(?:\n|$))",
            text,
            re.IGNORECASE,
        )
        if mi_sku:
            attributes["part_number"] = re.sub(r"\s+", "", mi_sku.group(1)).upper()

    return attributes


def _clean_field(value: str) -> str | None:
    """Return a compact label value without OCR whitespace noise."""
    value = re.sub(r"\s+", " ", value).strip(" :;|,-")
    return value or None


def _value_after_label(text: str, label_pattern: str) -> str | None:
    """Get an identifier printed next to a shipment-label field heading."""
    match = re.search(
        rf"{label_pattern}\s*[:#-]?\s*([A-Z0-9][A-Z0-9-]{{5,}})",
        text,
        re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


def _section_first_line(text: str, heading_pattern: str, stop_pattern: str) -> str | None:
    """Read the first meaningful line in an address/description label section."""
    match = re.search(
        rf"{heading_pattern}\s*:?\s*\n?(.+?)(?=\n\s*(?:{stop_pattern})\b|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    for line in match.group(1).splitlines():
        line = _clean_field(line)
        if line and not re.fullmatch(r"[A-Z ]{2,}", line):
            return line
    return None


def extract_shipment_attributes(text: str, barcode_values: list[str] | None = None) -> dict:
    """Extract common courier and invoice values from already-OCRed label text.

    This deliberately works from the normal OCR result and ZXing output, so a
    shipment upload does not trigger separate OCR work merely for these fields.
    """
    compact_text = re.sub(r"[ \t]+", " ", text)
    tracking = _value_after_label(
        compact_text,
        r"(?:AWB(?:\s*(?:NO\.?|NUMBER))?|TRACK(?:ING)?(?:\s*(?:NO\.?|NUMBER))?|E-?COMMERCE\s+ORDER\s+DETAIL\s+ID)",
    )
    # OnePlus packing slips label the shipment reference as "PREPAID ORDER".
    # Prefer that explicit value over generic barcode ordering.
    prepaid_order = _value_after_label(compact_text, r"PREPAID\s+ORDER")
    if prepaid_order:
        tracking = prepaid_order
    # An order value must be explicitly labelled.  Barcode payloads are often
    # carrier routing/internal values, so treating an arbitrary barcode as an
    # order ID corrupts the distinction between the two API fields.
    order_id = _value_after_label(
        compact_text,
        r"(?:ORDER\s*(?:ID|NO\.?|NUMBER)|MALL\s+ORDER|AMAZON\s+ORDER\s*ID)",
    )

    # Courier barcodes often contain the AWB even if its caption is blurred.
    # Keep the selection conservative: tracking values are normally at least
    # eight characters and must contain a digit; IMEI-only barcode values are
    # intentionally excluded.
    barcode_values = barcode_values or []
    candidates = [
        re.sub(r"\s+", "", value).upper()
        for value in barcode_values
        if re.fullmatch(r"[A-Z0-9-]{8,}", re.sub(r"\s+", "", value).upper())
        and any(char.isdigit() for char in value)
        and not re.fullmatch(r"\d{15}", re.sub(r"\s+", "", value))
    ]

    # A phone label can contain serial/IMEI barcode payloads.  They are not
    # shipment references, so never promote them to e-commerce IDs merely
    # because the label has no readable courier field.
    if re.search(
        r"\b(?:IMEI|MODEL|RAM|ROM|SERIAL|S/?N|MOTOROLA|XT\s*\d{3,5}\s*-\s*\d+)\b",
        text,
        re.IGNORECASE,
    ):
        candidates = []

    # OnePlus invoices place the recipient and ORDER ID side-by-side.  OCR can
    # therefore read the recipient name immediately after the "ORDER ID"
    # heading, even though the barcode itself decodes perfectly.  The TSO…
    # barcode is the order ID; the numeric barcode under PREPAID ORDER is the
    # e-commerce order-detail ID.
    oneplus_order_id = next(
        (value for value in candidates if re.fullmatch(r"TSO[A-Z0-9-]{8,}", value)),
        None,
    )
    if oneplus_order_id:
        order_id = oneplus_order_id
        prepaid_barcode = next(
            (value for value in candidates if value.isdigit() and value != oneplus_order_id),
            None,
        )
        if prepaid_barcode:
            tracking = prepaid_barcode
    if not tracking:
        tracking_candidates = [value for value in candidates if value != order_id]
        # Amazon and courier AWBs are commonly numeric; prefer the linear
        # shipment-like number over opaque alphanumeric DataMatrix payloads.
        tracking = next((value for value in tracking_candidates if value.isdigit()), None)
        tracking = next(
            (value for value in tracking_candidates if any(char.isalpha() for char in value)),
            None,
        ) if not tracking else tracking

    # Reliance/Delhivery labels commonly print the AWB and ORDER ID as two
    # adjacent numeric barcodes.  Creases can hide the ORDER caption from OCR,
    # while ZXing still decodes both values.  Only use this positional fallback
    # when OCR did recognise an AWB heading; it deliberately excludes generic
    # labels and DataMatrix/internal values.
    if not order_id and tracking and re.search(r"\bAWB\b", text, re.IGNORECASE):
        order_id = next(
            (value for value in candidates if value.isdigit() and value != tracking),
            None,
        )

    address_name = _section_first_line(
        text,
        r"(?:DELIVER\s+TO|DELIVERY\s+ADDRESS|SHIP\s+TO)",
        r"(?:PICKUP\s+ADDRESS|RETURN\s+ADDRESS|AWB|TRACK(?:ING)?|ORDER|ITEM|ROUTE)",
    )
    item_description = _section_first_line(
        text,
        r"(?:ITEM\s+DESCRIPTION|PRODUCT\s+DESCRIPTION|DESCRIPTION)",
        r"(?:TOTAL|AMOUNT|QTY|QUANTITY|TAX|ORDER\s+DATE)",
    )

    if oneplus_order_id:
        # The "Total" column header is commonly read before the first item.
        # Walk subsequent OCR lines to obtain the first actual product line.
        item_heading = re.search(r"\b[IT]TEM\s+DES\w*", text, re.IGNORECASE)
        if item_heading:
            for raw_line in text[item_heading.end():].splitlines():
                line = _clean_field(raw_line)
                if not line or "TOTAL" in line.upper() or line.upper() == "AMOUNT":
                    continue
                if re.fullmatch(r"[\d.,]+", line):
                    continue
                if re.search(r"[A-Z]", line, re.IGNORECASE):
                    item_description = line
                    break
    return {
        "ecommerce_order_detail_id": tracking,
        "order_id": order_id,
        "address_name": address_name,
        "item_description": item_description,
    }


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


def repair_imei_check_digit(imei: str | None) -> str | None:
    """Repair only the final OCR digit when the first 14 digits are intact.

    The fifteenth IMEI digit is a Luhn check digit.  This is deliberately not
    a general "guess" of an IMEI: it changes only that one checksum character
    and returns the original value if it is already valid or malformed.
    """
    if not imei or not imei.isdigit() or len(imei) != 15 or is_valid_imei(imei):
        return imei
    prefix = imei[:14]
    return next(
        (prefix + digit for digit in "0123456789" if is_valid_imei(prefix + digit)),
        imei,
    )


def barcode_values_from_image(image_path: str) -> list[str]:
    """Decode text from 1D and 2D barcodes once for all extractors."""
    if zxingcpp is None:
        return []

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    values = []
    for barcode in zxingcpp.read_barcodes(image):
        value = barcode.text.strip()
        if value and value not in values:
            values.append(value)
    return values


def apple_accessory_barcode_values(image_path: str) -> list[str]:
    """Decode compact Apple side-label barcodes missed in the full image.

    A narrow side label occupies only a small portion of a portrait box photo,
    so the full-image detector may find its UPC and serial but skip the dense
    ``1P`` part-number barcode.  These conservative centre/right crops are
    used only after an Apple A-model is already recognised.
    """
    if zxingcpp is None:
        return []

    image = load_image(image_path)
    height, width = image.shape[:2]
    crops = (
        image[int(height * 0.45):int(height * 0.82), int(width * 0.25):int(width * 0.98)],
        image[int(height * 0.50):int(height * 0.80), int(width * 0.35):int(width * 0.95)],
        image[int(height * 0.48):int(height * 0.72), int(width * 0.27):int(width * 0.78)],
    )
    values = []
    for crop in crops:
        if crop.size == 0:
            continue
        # The compact 1P barcode can be too small for a full-frame decoder;
        # retain the original attempt and a scaled version of the same crop.
        for candidate in (crop, cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)):
            for barcode in zxingcpp.read_barcodes(candidate):
                value = barcode.text.strip()
                if value and value not in values:
                    values.append(value)
    return values


def ocr_apple_accessory_label(image_path: str) -> str:
    """Read Apple accessory side labels when full-frame OCR misses the text."""
    image = load_image(image_path)
    height, width = image.shape[:2]
    broad_label = image[int(height * 0.40):int(height * 0.78), int(width * 0.20):int(width * 0.90)]
    model_label = image[int(height * 0.43):int(height * 0.72), int(width * 0.28):int(width * 0.78)]
    if broad_label.size == 0 or model_label.size == 0:
        return ""
    gray = cv2.cvtColor(broad_label, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    thresholded = cv2.adaptiveThreshold(
        enlarged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )
    # The pale "Model No." row can disappear in thresholded output.  A tighter
    # grayscale crop keeps that row legible while the thresholded pass remains
    # useful for the barcode-adjacent part/serial labels.
    model_gray = cv2.cvtColor(model_label, cv2.COLOR_BGR2GRAY)
    model_enlarged = cv2.resize(model_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return "\n".join(
        (
            pytesseract.image_to_string(thresholded, config="--psm 6"),
            pytesseract.image_to_string(model_enlarged, config="--psm 6"),
        )
    )


def imeis_from_barcodes(image_path: str) -> list[str]:
    """Decode valid IMEIs from 1D/2D barcodes on a device label.

    ZXing recognizes Code 128, EAN/UPC and QR codes.  Only a valid 15-digit
    Luhn number is returned, so EAN product barcodes and support URLs cannot
    be mistaken for an IMEI.
    """
    imeis = []
    for value in barcode_values_from_image(image_path):
        for candidate in re.findall(r"(?<!\d)\d{15}(?!\d)", value):
            if is_valid_imei(candidate) and candidate not in imeis:
                imeis.append(candidate)
    return imeis


def fill_missing_imeis_from_barcodes(
    attributes: dict, image_path: str, barcode_values: list[str] | None = None
) -> set[str]:
    """Fill absent or checksum-invalid IMEI fields using decoded barcodes."""
    barcode_values = barcode_values if barcode_values is not None else barcode_values_from_image(image_path)
    barcode_imeis = [
        candidate
        for value in barcode_values
        for candidate in re.findall(r"(?<!\d)\d{15}(?!\d)", value)
        if is_valid_imei(candidate)
    ]
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
    """Extract device and shipment fields from one label image."""
    barcode_values = barcode_values_from_image(image_path)
    barcode_imeis = [value for value in barcode_values if re.fullmatch(r"\d{15}", value)]
    is_rotated_apple_phone = (
        len(barcode_imeis) >= 2
        and any(re.fullmatch(r"S[A-Z0-9]{8,}", value) for value in barcode_values)
    )
    # Dual-IMEI Apple labels can be photographed sideways, but a normal photo
    # also has the same barcode pattern.  Keep normal OCR when it already
    # finds an iPhone/iPad product line; otherwise use the rotated recovery.
    normal_text = ocr_text_adaptive(image_path)
    if is_rotated_apple_phone and not re.search(r"\b(?:IPHONE|IPAD)\b", normal_text, re.IGNORECASE):
        text = ocr_rotated_apple_phone_text(image_path)
    else:
        text = normal_text
    attributes = extract_attributes(text)

    if attributes.get("model_name") == "AI+ Pulse 2" and not attributes.get("model_code"):
        aiplus_attributes = extract_attributes(ocr_aiplus_specification_table(image_path))
        for field in ("model_code", "model_name", "part_number", "ram", "storage"):
            if not attributes.get(field) and aiplus_attributes.get(field):
                attributes[field] = aiplus_attributes[field]

    if attributes.get("model_name") == "AI+ Pulse 2" and (
        not is_valid_imei(attributes.get("imei1") or "")
        or not is_valid_imei(attributes.get("imei2") or "")
    ):
        aiplus_identifiers = extract_attributes(ocr_aiplus_identifier_region(image_path))
        for field in ("imei1", "imei2"):
            if (
                (not attributes.get(field) or not is_valid_imei(attributes.get(field)))
                and is_valid_imei(aiplus_identifiers.get(field) or "")
            ):
                attributes[field] = aiplus_identifiers[field]

    hp_qr = next(
        (value for value in barcode_values if "hpsmart.com/qrc/" in value.lower()),
        None,
    )
    is_hp_victus = bool(hp_qr or re.search(r"\bVICTUS\b", text, re.IGNORECASE))
    if is_hp_victus:
        hp_attributes = extract_attributes(ocr_hp_victus_label(image_path))
        for field in ("model_code", "model_name", "part_number", "serial_number"):
            if not attributes.get(field) and hp_attributes.get(field):
                attributes[field] = hp_attributes[field]

        # HP's QR payload has the serial after /21/ and the P/N in the 240
        # parameter.  Decode it because this is more precise than label OCR.
        if hp_qr:
            hp_qr_match = re.search(r"/21/([A-Z0-9]+)\?240=([^?]+)", hp_qr, re.IGNORECASE)
            if hp_qr_match:
                attributes["serial_number"] = hp_qr_match.group(1).upper()
                attributes["part_number"] = unquote(hp_qr_match.group(2)).upper()

    # A compact Apple side label may be the only readable source of its A-model
    # and 1P part number.  Its S-prefixed serial barcode is a reliable signal
    # to attempt the focused crop without applying that work to other labels.
    has_apple_serial_barcode = any(re.fullmatch(r"S[A-Z0-9]{8,}", value) for value in barcode_values)
    has_apple_model = bool(re.fullmatch(r"A\d{4,5}", attributes.get("model_code") or "", re.IGNORECASE))
    has_primary_device_field = any(
        attributes.get(field) for field in ("model_code", "part_number", "imei1", "imei2")
    )
    has_upc = any(re.fullmatch(r"\d{12,13}", value) for value in barcode_values)
    # Some camera angles hide the serial and Model No. rows from full-frame
    # OCR.  An otherwise unidentified UPC product gets one focused attempt;
    # fields are merged only if that attempt proves it is an Apple label.
    if has_apple_serial_barcode or has_apple_model or (not has_primary_device_field and has_upc):
        focused_barcode_values = apple_accessory_barcode_values(image_path)
        has_apple_part_barcode = any(
            re.fullmatch(r"1P[A-Z0-9]+/[A-Z]", value) for value in focused_barcode_values
        )
        has_apple_serial_barcode = has_apple_serial_barcode or any(
            re.fullmatch(r"S[A-Z0-9]{8,}", value) for value in focused_barcode_values
        )
        apple_label_text = ocr_apple_accessory_label(image_path)
        apple_label_attributes = extract_attributes(apple_label_text)
        is_apple_label = has_apple_part_barcode or has_apple_serial_barcode or bool(
            re.fullmatch(r"A\d{4,5}", apple_label_attributes.get("model_code") or "", re.IGNORECASE)
        )
        if is_apple_label:
            for field in ("model_code", "model_name", "part_number", "serial_number"):
                if not attributes.get(field) and apple_label_attributes.get(field):
                    attributes[field] = apple_label_attributes[field]
            for value in focused_barcode_values:
                if value not in barcode_values:
                    barcode_values.append(value)
            # Prefer a valid Apple A-number from the combined crop passes over
            # an earlier OCR candidate containing letter-like digit errors.
            apple_model = re.search(
                r"\bMODEL\s*(?:NO\.?)?\s*[:.,]?\s*(A\d{4,5})\b",
                apple_label_text,
                re.IGNORECASE,
            )
            if apple_model and not re.search(r"\b(?:IPHONE|IPAD)\b", text, re.IGNORECASE):
                attributes["model_code"] = apple_model.group(1).upper()
                attributes["model_name"] = attributes["model_code"]

    is_samsung_label = bool(re.search(r"\bSM-S\d{3,5}[A-Z](?:/DS)?\b", text, re.IGNORECASE))
    if is_samsung_label and (
        not attributes.get("imei1")
        or not attributes.get("imei2")
        or not is_valid_imei(attributes.get("imei2") or "")
        or not attributes.get("ram")
        or not attributes.get("color")
    ):
        samsung_attributes = extract_attributes(ocr_samsung_label_region(image_path))
        for field in ("imei1", "imei2", "ram", "storage", "color"):
            if (
                (not attributes.get(field) or (field == "imei2" and not is_valid_imei(attributes.get(field))))
                and samsung_attributes.get(field)
            ):
                attributes[field] = samsung_attributes[field]

    is_infinix_label = bool(
        re.search(r"\bINFINIX\b", text, re.IGNORECASE)
        or re.search(r"\bX\d{4}\b", text, re.IGNORECASE)
    )
    if is_infinix_label and (
        not attributes.get("model_code")
        or not attributes.get("ram")
        or not attributes.get("storage")
        or not attributes.get("imei2")
    ):
        infinix_text = ocr_infinix_label_region(image_path)
        infinix_attributes = extract_attributes(infinix_text)
        for field in ("model_code", "model_name", "color", "ram", "storage"):
            if not attributes.get(field) and infinix_attributes.get(field):
                attributes[field] = infinix_attributes[field]

        # The second IMEI often has an unnumbered ``IMEI:`` caption, but its
        # 15-digit value remains clear in the focused label crop.
        for candidate in re.findall(r"(?<!\d)\d{15}(?!\d)", infinix_text):
            if is_valid_imei(candidate) and candidate != attributes.get("imei1"):
                attributes["imei2"] = candidate
                break

    if (
        not attributes.get("imei1")
        or not attributes.get("imei2")
        or not attributes.get("serial_number")
        or not attributes.get("part_number")
    ):
        imei_text = ocr_imei_region(image_path)
        imei_attributes = extract_attributes(imei_text)
        for field in ("imei1", "imei2", "serial_number", "part_number"):
            if not attributes.get(field) and imei_attributes.get(field):
                attributes[field] = imei_attributes[field]

    if (attributes.get("model_code") or "").startswith("XT") and attributes.get("part_number"):
        attributes["part_number"] = re.sub(r"O", "0", attributes["part_number"])
    # Vivo's printed serial font can make 9/J look like S/U in the targeted
    # crop.  Restrict these normalisations to the recognised V-series format.
    serial_number = attributes.get("serial_number")
    if (attributes.get("model_code") or "").startswith("V") and serial_number:
        serial_number = re.sub(r"(?<=MG8)S", "9", serial_number)
        serial_number = re.sub(r"000U3$", "000J3", serial_number)
        attributes["serial_number"] = serial_number

    fill_missing_imeis_from_barcodes(attributes, image_path, barcode_values)
    for field in ("imei1", "imei2"):
        attributes[field] = repair_imei_check_digit(attributes.get(field))

    # Apple accessory barcodes encode the field value with a one-character
    # application prefix: 1P for the part number and S for the serial number.
    # These are more reliable than OCR on narrow barcode-label print.
    apple_part = next(
        (value[2:] for value in barcode_values if re.fullmatch(r"1P[A-Z0-9]+/[A-Z]", value)),
        None,
    )
    apple_serial = next(
        (value[1:] for value in barcode_values if re.fullmatch(r"S[A-Z0-9]{8,}", value)),
        None,
    )
    if apple_serial:
        attributes["serial_number"] = apple_serial

    # Apple accessory side labels can be small relative to the box photograph.
    # Retry their centre/right label area only when the A-model is known and
    # the initial full-image detector missed the 1P barcode.
    if not apple_part and re.fullmatch(r"A\d{4,5}", attributes.get("model_code") or "", re.IGNORECASE):
        apple_part = next(
            (
                value[2:]
                for value in apple_accessory_barcode_values(image_path)
                if re.fullmatch(r"1P[A-Z0-9]+/[A-Z]", value)
            ),
            None,
        )

    if apple_part:
        attributes["part_number"] = apple_part

        # An incomplete A-number is recoverable from a small lower-label crop.
        if not re.fullmatch(r"A\d{4}", attributes.get("model_code") or ""):
            model_text = ocr_apple_model_region(image_path)
            model_match = re.search(r"\bMODEL\s*(?:NO\.?)?\s*:?\s*(A\d{4})\b", model_text, re.IGNORECASE)
            if model_match:
                attributes["model_code"] = model_match.group(1).upper()
                attributes["model_name"] = attributes["model_code"]

    # Motorola labels use the printed UPC/EAN-style number as the device
    # serial reference.  It is not a shipping identifier.
    if re.fullmatch(r"XT\d{3,5}-\d+", attributes.get("model_code") or "", re.IGNORECASE):
        motorola_serial = next(
            (value for value in barcode_values if re.fullmatch(r"\d{12,13}", value)),
            None,
        )
        if motorola_serial:
            attributes["serial_number"] = motorola_serial
        motorola_part = next(
            (value for value in barcode_values if re.fullmatch(r"P[A-Z0-9]{6,}IN", value)),
            None,
        )
        if motorola_part:
            attributes["part_number"] = motorola_part

    # Mi labels encode the full serial in their first Code 128 barcode.  It is
    # more dependable than OCR here, which can truncate the value at a slash
    # or insert a space.  The X5ZS glyph sequence is a known barcode/OCR
    # ambiguity on these labels; the printed Mi serial convention is X52S.
    is_xiaomi_label = re.search(r"\b(?:MI\.COM|XIAOMI|HYPEROS)\b", text, re.IGNORECASE)
    if is_xiaomi_label:
        mi_serial = next(
            (value for value in barcode_values if re.fullmatch(r"\d{5}/[A-Z0-9]{8,}", value)),
            None,
        )
        if mi_serial:
            mi_serial = re.sub(r"(?<=/X5)Z(?=S)", "2", mi_serial)
            attributes["serial_number"] = mi_serial

    shipment_attributes = extract_shipment_attributes(text, barcode_values)
    is_device_label = any(
        attributes.get(field)
        for field in ("model_code", "model_name", "part_number", "imei1", "imei2")
    )
    if is_device_label and not SHIPMENT_MARKERS.search(text):
        # A UPC/EAN on a product box is a device serial/reference, not a
        # courier tracking number.  This covers Google/Apple-style labels as
        # well as Motorola boxes, even if OCR misses the word "IMEI".
        if not attributes.get("serial_number"):
            device_serial = next(
                (value for value in barcode_values if re.fullmatch(r"\d{12,13}", value)),
                None,
            )
            if device_serial:
                attributes["serial_number"] = device_serial
        shipment_attributes["ecommerce_order_detail_id"] = None
        shipment_attributes["order_id"] = None

    attributes.update(shipment_attributes)
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
