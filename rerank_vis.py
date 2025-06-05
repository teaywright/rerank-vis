import streamlit as st
import json
from PIL import Image, ImageDraw
from pathlib import Path

st.set_page_config(layout="wide")
st.title("Gaze vs REC: Click to Show Path/Point")

# ─── 1) Load JSON files (must now contain "gaze_sequences" and "rec_points") ───
with open("gaze_scoring_results.json") as f:
    gaze_data = json.load(f)

with open("rec_scoring_results.json") as f:
    rec_data = json.load(f)

# Build a dict keyed by image_path.
data = {}
for g_entry, r_entry in zip(gaze_data, rec_data):
    image_path = g_entry["image_path"]
    assert image_path == r_entry["image_path"]

    # Pull the lists (or default to empty)
    gaze_sequences = g_entry.get("gaze_sequences", [])
    rec_points = r_entry.get("rec_points", [])

    # Ensure they’re lists of the correct length
    if not isinstance(gaze_sequences, list):
        gaze_sequences = []
    if not isinstance(rec_points, list):
        rec_points = []

    n_candidates = len(g_entry["candidates"])
    if len(gaze_sequences) != n_candidates:
        # If missing or mis‐aligned, replace with empty lists for each candidate
        gaze_sequences = [[] for _ in range(n_candidates)]
    if len(rec_points) != n_candidates:
        # If missing or mis‐aligned, replace with None for each candidate
        rec_points = [None for _ in range(n_candidates)]

    data[image_path] = {
        "bbox": g_entry["bbox"],
        "candidates": g_entry["candidates"],
        "gaze_distances": g_entry["gaze_distances"],
        "rec_distances": r_entry["rec_distances"],
        "gaze_sequences": gaze_sequences,
        "rec_points": rec_points,
    }

# ─── 2) Helper: load from local “data/overlayed_images/” ───────────────────────
def get_image_by_filename(filename: str) -> Image.Image | None:
    local_path = Path("output/overlayed_images") / filename
    if local_path.exists():
        return Image.open(local_path)
    return None

# ─── 3) Drawing functions ──────────────────────────────────────────────────────
def draw_bbox(im: Image.Image, bbox: list[float], color: str = "red", width: int = 3) -> Image.Image:
    canvas = im.copy()
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = bbox
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return canvas

def draw_gaze_path(im: Image.Image, path: list[list[float]], color: str = "lime", width: int = 2) -> Image.Image:
    """
    Given a list of [x, y] points, draw a polyline + small circles for each fixation.
    If path is empty or fewer than 2 points, return im unchanged.
    """
    if not path or len(path) < 2:
        return im
    canvas = im.copy()
    draw = ImageDraw.Draw(canvas)
    pts = [(pt[0], pt[1]) for pt in path]
    draw.line(pts, fill=color, width=width)
    for x, y in pts:
        r = 3
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=1)
    return canvas

def draw_rec_point(im: Image.Image, point: list[float], color: str = "yellow", radius: int = 5) -> Image.Image:
    """
    Given a single [x, y], draw a small filled circle. If point is None or invalid, return im unchanged.
    """
    if not point or len(point) != 2:
        return im
    x, y = point
    canvas = im.copy()
    draw = ImageDraw.Draw(canvas)
    r = radius
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)
    return canvas

# ─── 4) Streamlit UI ───────────────────────────────────────────────────────────
example_keys = list(data.keys())
selected_image_path = st.selectbox("🔎 Choose an example (image_path):", example_keys)

entry = data[selected_image_path]
filename = Path(selected_image_path).name
bbox = entry["bbox"]
candidates = entry["candidates"]
gaze_distances = entry["gaze_distances"]
rec_distances = entry["rec_distances"]
gaze_sequences = entry["gaze_sequences"]
rec_points = entry["rec_points"]

# 4.1 – Load the base image
base_image = get_image_by_filename(filename)
if base_image is None:
    st.error(f"Could not find '{filename}' in data/overlayed_images/.")
    st.stop()

# 4.2 – Draw bounding box once
img_with_bbox = draw_bbox(base_image, bbox, color="red", width=3)
st.subheader("① Base image + bounding box")
st.image(img_with_bbox, use_container_width=True)

# 4.3 – Track which candidate-button was clicked
if "selected_candidate_idx" not in st.session_state:
    st.session_state.selected_candidate_idx = None

# 4.4 – One button per candidate
st.subheader("② Click a candidate button to overlay its Gaze path & REC point")
cols = st.columns(2)
for idx, cand in enumerate(candidates):
    col = cols[idx % 2]
    btn_text = f"{cand['text']}  ({cand['type']})"
    if col.button(btn_text, key=f"btn_{idx}"):
        st.session_state.selected_candidate_idx = idx

# 4.5 – Redraw overlays for the clicked candidate
sel_idx = st.session_state.selected_candidate_idx
if sel_idx is not None:
    st.markdown("---")
    st.write(f"**Selected candidate:** {candidates[sel_idx]['text']}  — Type: `{candidates[sel_idx]['type']}`")

    # 4.5.1 – Start from the bounding-box image
    overlay = img_with_bbox.copy()

    # 4.5.2 – Overlay that candidate’s gaze sequence (if available)
    gaze_path = gaze_sequences[sel_idx] if (
        gaze_sequences and sel_idx < len(gaze_sequences)
    ) else []
    overlay = draw_gaze_path(overlay, gaze_path, color="lime", width=2)

    # 4.5.3 – Overlay that candidate’s REC point (if available)
    rec_pt = rec_points[sel_idx] if (
        rec_points and sel_idx < len(rec_points)
    ) else None
    overlay = draw_rec_point(overlay, rec_pt, color="yellow", radius=5)

    # 4.5.4 – Show the final overlay
    st.subheader("③ Overlay: Gaze path (lime) + REC point (yellow)")
    st.image(overlay, use_container_width=True)

# 4.6 – Always show the two reranked tables at the bottom
def reranked_table(cands, dists, title: str):
    st.write(f"#### {title}")
    sorted_items = sorted(zip(dists, cands), key=lambda x: x[0])
    for dist, cand in sorted_items:
        tag = "🟡 Gold" if cand["type"] == "gold" else "⚪️ Gen"
        st.markdown(f"- `{dist:.2f}`  **{cand['text']}** ({tag})")

st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    reranked_table(candidates, gaze_distances, "👁️ Gaze-Based Ranking")
with col_b:
    reranked_table(candidates, rec_distances, "📍 REC-Based Ranking")
