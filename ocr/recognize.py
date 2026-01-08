import torch
from config import MODEL_PATH, IDX2CHAR
from ocr.preprocess import preprocess_char
from ocr.model import EnhancedBMCNNwHFCs

device = torch.device("cpu")

# ✅ Load checkpoint dictionary
checkpoint = torch.load(MODEL_PATH, map_location=device)

# ✅ Rebuild model with SAME parameters
model = EnhancedBMCNNwHFCs(
    num_classes=len(IDX2CHAR),
    dropout_rate=checkpoint["config"]["dropout"]
).to(device)

# ✅ Load trained weights
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

@torch.no_grad()
def recognize(chars):
    text = ""
    for img in chars:
        x = preprocess_char(img).to(device)
        logits = model(x)
        idx = torch.argmax(logits, dim=1).item()
        text += IDX2CHAR[idx]
    return text
