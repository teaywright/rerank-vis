import streamlit as st
import json
from PIL import Image, ImageDraw
from pathlib import Path
from matplotlib import cm

st.set_page_config(layout="wide")
st.title("Gaze vs REC: Reranking + Detailed Overlays (Scaled to 512×320)")

# ─── 1) Load JSON files (containing "gaze_sequences" and "rec_points") ───────
with open("gaze_scoring_results.json") as f:
    gaze_data = json.load(f)

with open("rec_scoring_results.json") as f:
    rec_data = json.load(f)

# Build a dict keyed by image_path
data = {}
for g_entry, r_entry in zip(gaze_data, rec_data):
    image_path = g_entry["image_path"]
    assert image_path == r_entry["image_path"], "Mismatch between gaze and rec entries"

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

# ─── 3) Custom drawing for gaze sequences ─────────────────────────────────────
def draw_sequence(draw: ImageDraw.ImageDraw, seq: list[tuple[float, float]],
                  cmap_name: str = "viridis", last_color: str = "red", label: bool = True):
    """
    Given a list of (x_pixel, y_pixel) points (already scaled), draw:
      - Each fixation: a filled circle with color from a colormap
      - A thick colored line between consecutive fixations
      - A small index label next to each fixation
      - An outline around the final fixation in `last_color`
    """
    # Keep only valid (x,y) pairs
    clean_seq = [ (x, y) for (x,y) in seq 
                  if isinstance(x, (int, float)) and isinstance(y, (int, float)) ]
    if len(clean_seq) < 1:
        return

    N = len(clean_seq)
    cmap = cm.get_cmap(cmap_name, N)

    for i, (x, y) in enumerate(clean_seq):
        # Convert the colormap float → 0–255
        rgb = tuple(int(255 * c) for c in cmap(i)[:3])
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=rgb)
        if i > 0:
            x_prev, y_prev = clean_seq[i - 1]
            draw.line([(x_prev, y_prev), (x, y)], fill=rgb, width=4)
        if label:
            draw.text((x + 6, y - 6), str(i + 1), fill="white")

    # Outline the last fixation with a thicker circle
    fx, fy = clean_seq[-1]
    r2 = 6
    draw.ellipse((fx - r2, fy - r2, fx + r2, fy + r2), outline=last_color, width=3)

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

# 4.1 – Load the overlayed image (512×320 with bbox already drawn)
base_image = get_image_by_filename(filename)
if base_image is None:
    st.error(f"Could not find '{filename}' in output/overlayed_images/.")
    st.stop()

# Precompute scale factors from 0–100 → actual image pixels
img_w, img_h = base_image.size   # should be (512, 320)
sx = img_w / 100.0
sy = img_h / 100.0

# ────────────────────────────────────────────────────────────────────────────────
# 4.2 – Display the single overlayed image at the top
st.subheader("① Overlayed Image (with bounding box already)")
st.image(base_image, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# 4.3 – Show reranked tables side by side
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
# 4.4 – For each candidate, put an expander showing both overlays
st.markdown("## Detailed Overlays per Candidate  (Scroll ↓ to expand)")
for idx, cand in enumerate(candidates):
    with st.expander(f"{idx+1}. \"{cand['text']}\" – Type: `{cand['type']}`"):
        # 4.4.1 – Scale & draw gaze sequence
        raw_seq = gaze_sequences[idx] if idx < len(gaze_sequences) else []
        # Convert each (x,y) from [0..100] → [0..img_w], [0..img_h]
        scaled_seq: list[tuple[float,float]] = [
            (pt[0] * sx, pt[1] * sy)
            for pt in raw_seq
            if isinstance(pt, (list, tuple)) and len(pt) == 2
        ]
        img_gaze = base_image.copy()
        draw_g = ImageDraw.Draw(img_gaze)
        draw_sequence(draw_g, scaled_seq, cmap_name="viridis", last_color="red", label=True)

        # 4.4.2 – Scale & draw REC point
        raw_pt = rec_points[idx] if idx < len(rec_points) else None
        img_rec = base_image.copy()
        draw_r = ImageDraw.Draw(img_rec)
        if isinstance(raw_pt, (list, tuple)) and len(raw_pt) == 2:
            x0, y0 = raw_pt[0] * sx, raw_pt[1] * sy
            r = 6
            draw_r.ellipse((x0 - r, y0 - r, x0 + r, y0 + r),
                           fill="yellow", outline="yellow")

        st.write("**Gaze Path (colored by fixation order)**")
        st.image(img_gaze, use_container_width=True)

        st.write("**REC Point (yellow circle)**")
        st.image(img_rec, use_container_width=True)

        # Optionally show the numeric distances
        st.write(f"- Gaze distance: `{gaze_distances[idx]:.2f}`")
        st.write(f"- REC distance: `{rec_distances[idx]:.2f}`")
        st.write("---")
