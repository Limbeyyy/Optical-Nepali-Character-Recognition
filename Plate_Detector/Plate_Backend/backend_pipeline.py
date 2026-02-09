from typing import Dict, List, Tuple
from config import username, password
from Plate_Detector.Plate_Backend.api_calls import (
    get_kataho_data,
    get_kataho_token
)
from Plate_Detector.Plate_Backend.backend_utils import (
    extract_qr_id,
    normalize_kid,
    normalize_text,
    fields_match
)
from Plate_Detector.Plate_Backend.api_num_conversion import nepali_to_english_digits, normalize_kataho_address

# --------------------------------------------------------
# MAIN VALIDATION FUNCTION
# --------------------------------------------------------

def validate_ocr_with_api(
    ocr_results_rounds: List[Dict],
    previously_verified: Dict[str, str] = None
) -> Tuple[Dict, Dict, List[str]]:
    """
    Progressive field-by-field validation with API caching.

    Returns:
        final_output          → All verified values
        verified_fields       → Locked fields
        remaining_fields      → Still need OCR
    """

    all_fields = [
        "KID_No",
        "Kataho_Address",
        "Plus_Code",
        "QR_Code",
        "Local_Address",
        "Ward_Address",
        "City"
    ]

    final_output = {field: None for field in all_fields}
    verified_fields = previously_verified.copy() if previously_verified else {}

    # --------------------------------------------------------
    # Fetch token once
    # --------------------------------------------------------
    token = get_kataho_token(username, password)
    if not token:
        return final_output, verified_fields, all_fields

    # --------------------------------------------------------
    # API cache: store responses for each KID+QR combination
    # --------------------------------------------------------
    api_cache: Dict[str, Dict] = {}

    # --------------------------------------------------------
    # PROCESS EACH OCR ROUND
    # --------------------------------------------------------
    for round_res in ocr_results_rounds:

        kid = round_res.get("KID_No")
        qr = round_res.get("QR_Code")

        if not kid or not qr:
            continue

        qr_id = extract_qr_id(qr)
        if not qr_id:
            continue

        kid = normalize_kid(kid)

        # Use cache key as combination of KID + QR
        cache_key = qr_id

        if cache_key in api_cache:
            api_response = api_cache[cache_key]
        else:
            api_response = get_kataho_data(
                username=username,
                password=password,
                kid=kid,
                qr_text=qr
            )
            if not api_response:
                continue
            # Ensure KID matches to trust API
            # if normalize_text(api_response.get("KID_No")) != normalize_text(kid):
            #     continue
            api_cache[cache_key] = api_response

        # --------------------------------------------------------
        # FIELD-BY-FIELD MATCHING (NO THRESHOLD)
        # --------------------------------------------------------
        for field in all_fields:

            # Skip already verified
            if field in verified_fields:
                continue

            api_val = api_response.get(field)
            ocr_val = round_res.get(field)

            # Special handling for QR
            if field == "QR_Code":
                extracted_from_ocr = extract_qr_id(ocr_val)
                if extracted_from_ocr and extracted_from_ocr == qr_id:
                    verified_fields[field] = qr_id
                continue

            # Special handling for Local_Address & Ward_Address → convert OCR Nepali digits → English
            if field in ["Kataho_Address"]:
                if ocr_val:
                    api_val_eng = normalize_kataho_address(api_val)
                    ocr_val_eng = normalize_kataho_address(ocr_val)
                    
                    if fields_match(api_val_eng, ocr_val_eng):
                        verified_fields[field] = ocr_val
                continue

            if field in ["Ward_Address"]:
                if ocr_val:
                    
                    print(api_val)
                    
                    print(ocr_val)
                    if fields_match(api_val, ocr_val):
                        verified_fields[field] = ocr_val
                continue

            # Normal field comparison
            if fields_match(api_val, ocr_val):
                verified_fields[field] = api_val

        # Stop early if everything verified
        if len(verified_fields) == len(all_fields):
            break

    # --------------------------------------------------------
    # Prepare outputs
    # --------------------------------------------------------
    final_output.update(verified_fields)

    remaining_fields = [
        f for f in all_fields if f not in verified_fields
    ]

    return final_output, verified_fields, remaining_fields
