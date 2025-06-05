import streamlit as st
import json
from PIL import Image, ImageDraw
from pathlib import Path

st.set_page_config(layout="wide")
st.title("Gaze vs REC: Reranking + Detailed Overlays")

# ─── 1) Load JSON files (now containing "gaze_sequences" and "rec_points") ───────
with open("gaze_scoring_results.json") as f:
    gaze_data = json.load(f)

with open("rec_scoring_results.json") as f:
    rec_data = json.load(f)

# Build a dict keyed by image_path
data = {}
for g_entry, r_entry in zip(gaze_data, rec_data):
    image_path = g_entry["image_path"]
    assert image_path == r_entry["image_path"], "Mismatch between gaze and rec entries"

    # Pull the lists (or default to empty)
    gaze_sequences = g_entry.get("gaze_sequences", [])
    rec_points      = r_entry.get("rec_points", [])

    # Ensure they’re lists of the correct length
    n_candidates = len(g_entry["candidates"])
    if not isinstance(gaze_sequences, list) or len(gaze_sequences) != n_candidates:
        gaze_sequences = [[] for _ in range(n_candidates)]
    if not isinstance(rec_points, list) or len(rec_points) != n_candidates:
        rec_points = [None for _ in range(n_candidates)]

    data[image_path] = {
        "candidates": g_entry["candidates"],
        "gaze_distances": g_entry["gaze_distances"],
        "rec_distances": r_entry["rec_distances"],
        "gaze_sequences": gaze_sequences,
        "rec_points": rec_points,
    }

# ─── 2) Helper: load from local “output/overlayed_images/” ────────────────────
def get_image_by_filename(filename: str) -> Image.Image | None:
    local_path = Path("output/overlayed_images") / filename
    if local_path.exists():
        return Image.open(local_path)
    return None

# ─── 3) Drawing functions ──────────────────────────────────────────────────────
def draw_gaze_path(im: Image.Image, path: list[list[float]], color: str = "lime", width: int = 2) -> Image.Image:
    """
    Given a list of [x, y] points, draw a polyline + small circles for each fixation.
    If path is empty or fewer than 2 points, return im unchanged.
    """
    if not path or not isinstance(path, list) or len(path) < 2:
        return im
    canvas = im.copy()
    draw = ImageDraw.Draw(canvas)
    pts = [(pt[0], pt[1]) for pt in path if isinstance(pt, list) and len(pt) == 2]
    if len(pts) < 2:
        return im
    draw.line(pts, fill=color, width=width)
    for x, y in pts:
        r = 3
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=1)
    return canvas

def draw_rec_point(im: Image.Image, point: list[float], color: str = "yellow", radius: int = 5) -> Image.Image:
    """
    Given a single [x, y], draw a small filled circle. If point is None or invalid, return im unchanged.
    """
    if not point or not isinstance(point, list) or len(point) != 2:
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
candidates = entry["candidates"]
gaze_distances = entry["gaze_distances"]
rec_distances = entry["rec_distances"]
gaze_sequences = entry["gaze_sequences"]
rec_points      = entry["rec_points"]

# 4.1 – Load the overlayed image (already has bbox drawn)
base_image = get_image_by_filename(filename)
if base_image is None:
    st.error(f"Could not find '{filename}' in output/overlayed_images/.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# 4.2 – Show reranked tables side by side
def reranked_table(cands, dists, title: str):
    st.write(f"#### {title}")
    sorted_items = sorted(zip(dists, cands), key=lambda x: x[0])
    for dist, cand in sorted_items:
        tag = "🟡 Gold" if cand["type"] == "gold" else "⚪️ Gen"
        st.markdown(f"- `{dist:.2f}`  **{cand['text']}** ({tag})")

st.markdown("## Reranked Lists")
col_a, col_b = st.columns(2)
with col_a:
    reranked_table(candidates, gaze_distances, "👁️ Gaze-Based Ranking")
with col_b:
    reranked_table(candidates, rec_distances, "📍 REC-Based Ranking")

# ────────────────────────────────────────────────────────────────────────────────
# 4.3 – For each candidate, put an expander showing both overlays
st.markdown("## Detailed Overlays per Candidate  (Scroll ↓ to expand)")
for idx, cand in enumerate(candidates):
    with st.expander(f"{idx+1}. \"{cand['text']}\" – Type: `{cand['type']}`"):
        # Draw gaze path overlay
        gaze_path = gaze_sequences[idx] if idx < len(gaze_sequences) else []
        img_gaze = draw_gaze_path(base_image, gaze_path, color="lime", width=2)

        # Draw REC point overlay (on the same base image)
        rec_pt = rec_points[idx] if idx < len(rec_points) else None
        img_rec = draw_rec_point(base_image, rec_pt, color="yellow", radius=5)

        st.write("**Gaze Path (lime)**")
        st.image(img_gaze, use_container_width=True)
        st.write("**REC Point (yellow)**")
        st.image(img_rec, use_container_width=True)

        # Optionally also display the raw distance values:
        st.write(f"- Gaze distance: `{gaze_distances[idx]:.2f}`")
        st.write(f"- REC distance: `{rec_distances[idx]:.2f}`")
        st.write("---")
