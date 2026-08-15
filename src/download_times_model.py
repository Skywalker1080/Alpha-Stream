import os
from pathlib import Path
import timesfm



MODEL_ID = "google/timesfm-2.5-200m-pytorch"
OUT_DIR = Path("model") / "timesfm-2.5-200m-pytorch"

OUT_DIR.mkdir(parents=True, exist_ok=True)

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
model.save_pretrained(OUT_DIR)

print(f"Saved to: {OUT_DIR.resolve()}")

