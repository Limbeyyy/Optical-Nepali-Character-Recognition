import re           
import requests     
import cv2          
import time         
import os           
import json         
from typing import Dict, List
from config import username, password, LOGIN_URL, DATA_URL
from Plate_Detector.Plate_Backend.backend_utils import extract_qr_id


def get_kataho_data(username: str, password: str, kid: str, qr_text: str):
    """
    Logs in → calls plate-status-check with BOTH KID and QR
    Returns cleaned OCR result dict
    """

    # ---------- LOGIN ----------
    login_url = LOGIN_URL
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
            print(" Token missing")
            return None
    except Exception as e:
        print(f" Login failed: {e}")
        return None

    # ---------- EXTRACT QR ID ----------
    qr_id = extract_qr_id(qr_text)
    if not kid or not qr_id:
        print(" KID or QR missing")
        return None

    # ---------- FETCH DATA ----------
    data_url = DATA_URL
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
        print(f" Data fetch failed: {e}")
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
    login_url = LOGIN_URL  # Use the configured login URL from config.py
    payload = {
        "username": username,   
        "password": password,
        "device_token": "abcd" #or any random string, as it's not used for actual device management in this context
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(login_url, json=payload, headers=headers, timeout=5)
        response.raise_for_status()

        data = response.json()
        # Usually the token is under "api_token" for this api
        token = data.get("data", {}).get("api_token")
        if token:
            print("Token retrieved successfully")
            return token
        else:
            print(" Token not found in response:", data)
            return None

    except requests.exceptions.Timeout:
        print("Login API timeout")
        return None
    except requests.exceptions.RequestException as e:
        print(f" Login API failed: {e}")
        return None
    except ValueError:
        print(" Login API returned non-JSON response")
        return None
