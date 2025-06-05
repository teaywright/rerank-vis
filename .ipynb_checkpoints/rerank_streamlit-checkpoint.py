import streamlit as st
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
import matplotlib.cm as cm

# --- CONFIG ---
RERANK_PATH = Path("rerank_results_10.json")
HF_DATASET  = "lmms-lab/RefCOCO"
HF_SPLIT    = "val"
TARGET_SIZE = (512, 320)
DISPLAY_W   = 600

# --- Load HF dataset images once ---
@st.cache_data
def load_refcoco_images():
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    return {ex["file_name"]: ex["image"].convert("RGB") for ex in ds}

hf_images = load_refcoco_images()

def pad_to_target(img: Image.Image, size=TARGET_SIZE, color=(255,255,255)):
    W, H = size
    ow, oh = img.size
    scale = min(W/ow, H/oh)
    nw, nh = int(ow*scale), int(oh*scale)
    resized = img.resize((nw, nh), Image.BICUBIC)
    out = Image.new("RGB", size, color)
    x, y = (W-nw)//2, (H-nh)//2
    out.paste(resized, (x, y))
    return out, x, y

def draw_sequence(draw, seq, cmap_name="viridis", last_color="red", label=True):
    clean_seq = [tuple(pt) for pt in seq if isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt)]
    if not clean_seq:
        return

    N = len(clean_seq)
    cmap = cm.get_cmap(cmap_name, N)

    for i, (x, y) in enumerate(clean_seq):
        color = tuple(int(255 * c) for c in cmap(i)[:3])
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        if i > 0:
            draw.line([clean_seq[i - 1], (x, y)], fill=color, width=4)  # thicker line
        if label:
            draw.text((x + 5, y - 5), str(i + 1), fill="white")

    fx, fy = clean_seq[-1]
    draw.ellipse((fx - 6, fy - 6, fx + 6, fy + 6), outline=last_color, width=3)

@st.cache_data
def load_rerank(path: Path):
    return json.loads(path.read_text())

data = load_rerank(RERANK_PATH)

st.title("Reranking Visualization (REC vs Gaze)")

font = ImageFont.load_default()

for item in data:
    fname = item["imagefile"]
    st.header(fname)
    if fname not in hf_images:
        st.error(f"Missing HF image for {fname}")
        continue

    base, dx, dy = pad_to_target(hf_images[fname])

    for title, key in [("REC Ranking", "order_rec"), ("Gaze Ranking", "order_gaze")]:
        st.subheader(title)
        order = item[key]
        sequences = item["denorm_gaze_sequences"]
        candidates = item["all_candidates"]
    
        for rank_idx, cand_idx in enumerate(order):
            label = candidates[cand_idx]
            st.markdown(f"**{rank_idx+1}. {label}**")
    
            if key == "order_gaze":
                seq = sequences[cand_idx]
                img = base.copy()
                draw = ImageDraw.Draw(img)
                draw_sequence(draw, seq, cmap_name="plasma")
                st.image(img, width=DISPLAY_W)
