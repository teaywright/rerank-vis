import streamlit as st
import json
from PIL import Image, ImageDraw
from pathlib import Path
from matplotlib import cm

st.title("Gaze vs REC: Reranking + Detailed Overlays")
#st.image(base_image, width=600)



# ─── 1) Load JSON files (containing "gaze_sequences" with possible nulls) ───────
with open("output/gaze_scoring_results_new.json") as f:
    gaze_data = json.load(f)

with open("output/rec_scoring_results_new.json") as f:
    rec_data = json.load(f)

# Build a dict keyed by image_path
data = {}
for g_entry, r_entry in zip(gaze_data, rec_data):
    image_path = g_entry["image_path"]
    assert image_path == r_entry["image_path"], "Mismatch between gaze and rec entries"

    raw_gaze_sequences = g_entry.get("gaze_sequences", [])
    raw_rec_points     = r_entry.get("rec_points", [])

    n_candidates = len(g_entry["candidates"])
    # If lengths don’t match, pad with empties/None
    if not isinstance(raw_gaze_sequences, list) or len(raw_gaze_sequences) != n_candidates:
        raw_gaze_sequences = [[] for _ in range(n_candidates)]
    if not isinstance(raw_rec_points, list) or len(raw_rec_points) != n_candidates:
        raw_rec_points = [None for _ in range(n_candidates)]

    # Now store
    data[image_path] = {
        "candidates":     g_entry["candidates"],
        "gaze_distances": g_entry["gaze_distances"],
        "rec_distances":  r_entry["rec_distances"],
        "gaze_sequences": raw_gaze_sequences,
        "rec_points":     raw_rec_points,
    }

# ─── 2) Helper: load from local “output/overlayed_images/” ────────────────────
def get_image_by_filename(filename: str) -> Image.Image | None:
    p = Path("output/overlayed_images") / filename
    return Image.open(p) if p.exists() else None

# ─── 3) Drawing functions ──────────────────────────────────────────────────────
def draw_sequence(
    draw: ImageDraw.ImageDraw,
    seq_100: list[tuple[float, float]],
    sx: float,
    sy: float,
    cmap_name: str = "viridis",
    last_color: str = "red",
    label: bool = True
):
    """
    seq_100: list of (x,y) pairs in 0–100 coordinates (floats)
    sx, sy: scale factors to convert 0–100 → pixel coords
    This function:
      - Filters out any invalid or null entries
      - Scales each (x, y) by (sx, sy)
      - Draws a colored circle for each fixation (colormap by index)
      - Draws a thick line between consecutive fixations
      - Labels each fixation with its 1-based index
      - Outlines the final fixation in `last_color`
    """
    # 3.1) Filter & scale
    scaled_pts: list[tuple[int, int]] = []
    for pt in seq_100:
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            x100, y100 = pt
            if isinstance(x100, (int, float)) and isinstance(y100, (int, float)):
                xp = int(round(x100 * sx))
                yp = int(round(y100 * sy))
                scaled_pts.append((xp, yp))

    if len(scaled_pts) < 1:
        print("EMPTY GAZE PT LIST")
        return

    N = len(scaled_pts)
    cmap = cm.get_cmap(cmap_name, N)

    # 3.2) Draw each fixation & line
    for i, (x, y) in enumerate(scaled_pts):
        rgb = tuple(int(255 * c) for c in cmap(i)[:3])
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=rgb)
        if i > 0:
            x_prev, y_prev = scaled_pts[i - 1]
            draw.line([(x_prev, y_prev), (x, y)], fill=rgb, width=4)
        if label:
            draw.text((x + 5, y - 5), str(i + 1), fill="white")

    # 3.3) Outline the last fixation
    fx, fy = scaled_pts[-1]
    r2 = 6
    draw.ellipse((fx - r2, fy - r2, fx + r2, fy + r2), outline=last_color, width=3)

def draw_rec_point(
    draw: ImageDraw.ImageDraw,
    pt_100: list[float],
    sx: float,
    sy: float,
    color: str = "yellow",
    radius: int = 6
):
    """
    pt_100: single [x,y] in 0–100 coords
    sx, sy: scale factors to convert 0–100 → pixels
    Draws a filled circle at the scaled location.
    """
    if not isinstance(pt_100, (list, tuple)) or len(pt_100) != 2:
        return
    x100, y100 = pt_100
    if not isinstance(x100, (int, float)) or not isinstance(y100, (int, float)):
        return

    x_px = int(round(x100 * sx))
    y_px = int(round(y100 * sy))
    r = radius
    draw.ellipse((x_px - r, y_px - r, x_px + r, y_px + r), fill=color, outline=color)

# ─── 4) Streamlit UI ───────────────────────────────────────────────────────────
# Build readable labels using the first gold reference from gaze reranking
example_labels = []
example_keys = []

for img_path, entry in data.items():
    candidates = entry["candidates"]
    gaze_distances = entry["gaze_distances"]

    # Sort by gaze distance
    sorted_indices = sorted(range(len(candidates)), key=lambda i: gaze_distances[i])
    gold_text = None
    for i in sorted_indices:
        if candidates[i]["type"] == "gold":
            gold_text = candidates[i]["text"]
            break
    label = gold_text if gold_text else "[No Gold]"

    # Create a display label and keep mapping to real path
    example_labels.append(f"{label}")
    example_keys.append(img_path)
    
selected_label = st.selectbox("🔎 Choose an example:", example_labels)
selected_image_path = example_keys[example_labels.index(selected_label)]

entry = data[selected_image_path]
filename = Path(selected_image_path).name
candidates = entry["candidates"]
gaze_distances = entry["gaze_distances"]
rec_distances = entry["rec_distances"]
gaze_sequences = entry["gaze_sequences"]
rec_points     = entry["rec_points"]

# 4.1 – Load the overlayed image (512×320, with bbox already drawn)
base_image = get_image_by_filename(filename)
if base_image is None:
    st.error(f"Could not find '{filename}' in output/overlayed_images/.")
    st.stop()

img_w, img_h = base_image.size  # expected (512, 320)
sx = img_w / 100.0
sy = img_h / 100.0

# ────────────────────────────────────────────────────────────────────────────────
# 4.2 – Show the single overlayed image at the top
st.subheader("① Overlayed Image (with bounding box already)")
st.image(base_image, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# 4.3 – Show reranked tables side by side
def reranked_table(cands, dists, title: str):
    st.write(f"#### {title}")
    for dist, cand in sorted(zip(dists, cands), key=lambda x: x[0]):
        tag = "🟡 Gold" if cand["type"] == "gold" else "⚪️ Gen"
        st.markdown(f"- `{dist:.2f}`  **{cand['text']}** ({tag})")

st.markdown("## Reranked Lists")
col1, col2 = st.columns(2)
with col1:
    reranked_table(candidates, gaze_distances, "👁️ Gaze-Based Ranking")
with col2:
    reranked_table(candidates, rec_distances, "📍 REC-Based Ranking")

# ────────────────────────────────────────────────────────────────────────────────
# 4.4 – Detailed Overlays in Gaze‐ranked order (lowest distance first)
st.markdown("## Detailed Overlays per Candidate (Gaze‐ranked order)")

# Build a list of indices sorted by gaze_distance
gaze_sorted_indices = [i for _, i in sorted((d, i) for i, d in enumerate(gaze_distances))]

for idx in gaze_sorted_indices:
    cand = candidates[idx]
    st.write(f"**Reference:** {cand['text']}  (`{cand['type']}`)")

    # 4.4.1 – Draw gaze path (filtering out nulls)
    raw_seq = gaze_sequences[idx] if idx < len(gaze_sequences) else []
    img_gaze = base_image.copy()
    draw_g   = ImageDraw.Draw(img_gaze)
    draw_sequence(draw_g, raw_seq, sx, sy, cmap_name="viridis", last_color="red", label=True)

    # 4.4.2 – Draw REC point (if present)
    raw_rec = rec_points[idx] if idx < len(rec_points) else None
    img_rec = base_image.copy()
    draw_r  = ImageDraw.Draw(img_rec)
    draw_rec_point(draw_r, raw_rec, sx, sy, color="yellow", radius=6)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_gaze, use_container_width=True)
        st.write(f"- Gaze distance: `{gaze_distances[idx]:.2f}`")
    with col2:
        st.image(img_rec, use_container_width=True)
        st.write(f"- REC distance: `{rec_distances[idx]:.2f}`")

    st.write("---")
import pandas as pd

with st.sidebar:
    st.write({
        "total_examples": 100,
        "gold_at_top_rec": 36,
        "gold_at_top_gaze": 32,
        "gold_at_top_combined": 36,
        "gold_in_top3_rec": 57,
        "gold_in_top3_gaze": 71,
        "gold_in_top3_combined": 74,
        "avg_gold_position_rec": 5.8901098901098905,
        "avg_gold_position_gaze": 5.4322344322344325,
        "avg_gold_position_combined": 5.2967032967032965,
        "avg_distance_to_center_rec": 67.3057957222177,
        "avg_distance_to_center_gaze": 80.43525600992326,
        "avg_distance_to_center_combined": 73.8705258661,
})



