import re           
import requests     
import cv2          
import time         
import os           
import json         
from typing import Dict, List
from config import username, password

def extract_qr_id(qr_text):
    """
    Extracts the unique code from QR URL.
    Example: 'https://kataho.app/c/2NHIQW8Q1K0' -> '2NHIQW8Q1K0'
    """
    match = re.search(r'/c/([A-Za-z0-9]+)', qr_text)
    return match.group(1) if match else None


import requests
import re

def normalize_kid(kid_text: str):
    if not kid_text:
        return None
    return kid_text.replace("KID:", "").strip()


def get_kataho_data(username: str, password: str, kid: str, qr_text: str):
    """
    Logs in → calls plate-status-check with BOTH KID and QR
    Returns cleaned OCR result dict
    """

    # ---------- LOGIN ----------
    login_url = "https://kataho.app/api/login"
    login_payload = {
        "username": username,
        "password": password,
        "device_token": "ocr_device"
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        login_resp = requests.post(login_url, json=login_payload, headers=headers, timeout=5)
        login_resp.raise_for_status()
        token = login_resp.json().get("data", {}).get("api_token")
        if not token:
            print("❌ Token missing")
            return None
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return None

    # ---------- EXTRACT QR ID ----------
    qr_id = extract_qr_id(qr_text)
    if not kid or not qr_id:
        print("❌ KID or QR missing")
        return None

    # ---------- FETCH DATA ----------
    data_url = "https://kataho.app/api/plate-status-check"
    data_headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    params = {
        "kataho_code": kid,     # ← KID goes here
        "kid_code": qr_id       # ← QR extracted ID goes here
    }


    try:
        resp = requests.get(data_url, headers=data_headers, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        return None

    # ---------- PARSE RESULT ----------
    if not data.get("success"):
        return None

    ocr = data.get("ocr_result", {})
    if not ocr:
        return None

    return {
        "Local_Address": ocr.get("Local_Address"),
        "Kataho_Address": ocr.get("Kataho_Address"),
        "KID_No": ocr.get("KID_No", "").replace("KID:", "").strip(),
        "Plus_Code": ocr.get("Plus_Code"),
        "Ward_Address": ocr.get("Ward_Address"),
        "City": ocr.get("City"),
        "QR_Code": ocr.get("QR_Code"),
    }



def get_kataho_token(username: str, password: str):
    """
    Logs in to Kataho API and retrieves the Bearer token.
    Returns token string if successful, else None.
    """
    login_url = "https://kataho.app/api/login"  # replace with the real login endpoint
    payload = {
        "username": username,   # or "username" if API expects username
        "password": password,
        "device_token": "abcd"
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(login_url, json=payload, headers=headers, timeout=5)
        response.raise_for_status()

        data = response.json()
        # Usually the token is under "token" or "access_token"
        token = data.get("data", {}).get("api_token")
        if token:
            print("✅ Token retrieved successfully")
            return token
        else:
            print("⚠️ Token not found in response:", data)
            return None

    except requests.exceptions.Timeout:
        print("⚠️ Login API timeout")
        return None
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Login API failed: {e}")
        return None
    except ValueError:
        print("⚠️ Login API returned non-JSON response")
        return None


def validate_ocr_with_api(ocr_results_rounds):
    final_output = {
        "KID_No": None,
        "Kataho_Address": None,
        "Plus_Code": None,
        "QR_Code": None,
        "Local_Address": None,
        "Ward_Address": None,
        "City": None
    }

    token = get_kataho_token(username, password)
    if not token:
        return final_output

    headers = {"Authorization": f"Bearer {token}"}

    for round_res in ocr_results_rounds:
        kid = round_res.get("KID_No")
        qr = round_res.get("QR_Code")

        if not kid or not qr:
            continue

        qr_id = extract_qr_id(qr)
        if not qr_id:
            continue

        # 🔹 CALL PRIVATE API (example)
        api_response = get_kataho_data(
                        username=username,
                        password=password,
                        kid=kid,
                        qr_text=qr
                    )


        if not api_response:
            continue

        # ✅ KID must match
        if api_response.get("KID_No") != kid:
            continue

        # ✅ SUCCESS → TRUST API
        final_output.update({
            "KID_No": api_response.get("KID_No"),
            "Kataho_Address": api_response.get("Kataho_Address"),
            "Plus_Code": api_response.get("Plus_Code"),
            "QR_Code": qr_id,
            "Local_Address": api_response.get("Local_Address"),
            "Ward_Address": api_response.get("Ward_Address"),
            "City": api_response.get("City")
        })

        return final_output  # stop after first valid match

    return final_output
