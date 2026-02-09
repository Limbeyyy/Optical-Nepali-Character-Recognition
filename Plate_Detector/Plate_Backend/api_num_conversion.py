# -----------------------------------------
# English ↔ Nepali Digit Mapping
# -----------------------------------------
ENG_TO_NEP_DIGITS = str.maketrans({
    "0": "०","1": "१","2": "२","3": "३","4": "४",
    "5": "५","6": "६","7": "७","8": "८","9": "९",
})

NEP_TO_ENG_DIGITS = str.maketrans({
    "०": "0","१": "1","२": "2","३": "3","४": "4",
    "५": "5","६": "6","७": "7","८": "8","९": "9",
})

def english_to_nepali_digits(text: str) -> str:
    if not text:
        return text
    return text.translate(ENG_TO_NEP_DIGITS)

def nepali_to_english_digits(text: str) -> str:
    if not text:
        return text
    return text.translate(NEP_TO_ENG_DIGITS)


import re
def normalize_kataho_address(text: str) -> str:
    """
    Normalizes Kataho_Address for comparison:
    - Converts Nepali digits to English digits
    - Strips extra spaces
    """
    if not text:
        return ""
    
    # Convert Nepali digits to English
    text = text.translate(NEP_TO_ENG_DIGITS)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
