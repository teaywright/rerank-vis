# import streamlit as st
# import json
# from PIL import Image, ImageDraw
# from pathlib import Path

# # --- Load JSON files ---
# with open("gaze_scoring_results.json") as f:
#     gaze_data = json.load(f)

# with open("rec_scoring_results.json") as f:
#     rec_data = json.load(f)

# # --- Index by image_path ---
# data = {}
# for g_entry, r_entry in zip(gaze_data, rec_data):
#     image_path = g_entry["image_path"]
#     assert image_path == r_entry["image_path"]
#     data[image_path] = {
#         "bbox": g_entry["bbox"],
#         "candidates": g_entry["candidates"],
#         "gaze_distances": g_entry["gaze_distances"],
#         "rec_distances": r_entry["rec_distances"]
#     }

# # --- Load images from local data/ directory ---
# def get_image_by_filename(filename):
#     local_path = Path("data/overlayed_images/") / filename
#     if local_path.exists():
#         return Image.open(local_path)
#     return None

# # --- Streamlit UI ---
# st.title("Gaze vs REC Candidate Scoring Viewer")

# example_keys = list(data.keys())
# selected_key = st.selectbox("Choose an example:", example_keys)

# entry = data[selected_key]
# filename = Path(selected_key).name
# bbox = entry["bbox"]
# candidates = entry["candidates"]
# gaze_scores = entry["gaze_distances"]
# rec_scores = entry["rec_distances"]

# # --- Load Image ---
# image = get_image_by_filename(filename)
# if image is None:
#     st.error(f"Image '{filename}' not found in local data/ directory.")
#     st.stop()

# # # --- Draw BBox on Image ---
# # def draw_bbox(image, bbox, color="red", width=4):
# #     draw = ImageDraw.Draw(image)
# #     draw.rectangle(bbox, outline=color, width=width)
# #     return image

# st.subheader("Image with Bounding Box")
# #img_copy = image.copy()
# #img_with_bbox = draw_bbox(img_copy, bbox)
# st.image(image, caption=filename)

# # --- Create sortable ranking tables ---
# def reranked_table(candidates, distances, title):
#     sorted_items = sorted(zip(distances, candidates), key=lambda x: x[0])
#     st.subheader(title)
#     for dist, cand in sorted_items:
#         tag = "🟡 Gold" if cand["type"] == "gold" else "⚪️ Gen"
#         st.markdown(f"- `{dist:.2f}`: **{cand['text']}** ({tag})", unsafe_allow_html=True)

# col1, col2 = st.columns(2)
# with col1:
#     reranked_table(candidates, gaze_scores, "👁️ Gaze-Based Ranking")

# with col2:
#     reranked_table(candidates, rec_scores, "📍 REC-Based Ranking")

import streamlit as st
import json
from PIL import Image, ImageDraw
from pathlib import Path

st.set_page_config(layout="wide")
st.title("Gaze vs REC: On-Demand Visualization")

# ─── 1) Load your JSONs ────────────────────────────────────────────────────────
with open("gaze_scoring_results.json") as f:
    gaze_data = json.load(f)

with open("rec_scoring_results.json") as f:
    rec_data = json.load(f)

# Build a dict keyed by image_path
data = {}
for g_entry, r_entry in zip(gaze_data, rec_data):
    image_path = g_entry["image_path"]
    assert image_path == r_entry["image_path"]
    # NOTE: We assume (for illustration) that each g_entry/r_entry also contains
    #       actual coordinate data. 
    #       Here, I’m stubbing in two new fields:
    #         - "gaze_coords": a list of [x,y] pairs (the full gaze trajectory)
    #         - "rec_point": a single [x,y] (the predicted REC point)
    #
    #       In practice, replace these stubs with your real arrays. 
    #
    #    e.g. g_entry["gaze_coords"] might be [[x0, y0], [x1, y1], …]
    #         r_entry["rec_point"] might be [x_rec, y_rec]
    #
    # For now, I’ll just put empty placeholders so the code runs without errors.
    #
    g_entry.setdefault("gaze_coords", [])    # ← replace with real sequence
    r_entry.setdefault("rec_point", None)     # ← replace with a real [x,y]
    #
    data[image_path] = {
        "bbox": g_entry["bbox"],
        "candidates": g_entry["candidates"],
        "gaze_distances": g_entry["gaze_distances"],
        "rec_distances": r_entry["rec_distances"],
        "gaze_coords": g_entry["gaze_coords"],
        "rec_point": r_entry["rec_point"],
    }

# ─── 2) Helper: load from local “data/overlayed_images/” ───────────────────────
def get_image_by_filename(filename: str) -> Image.Image | None:
    local_path = Path("data/overlayed_images") / filename
    if local_path.exists():
        return Image.open(local_path)
    return None

# ─── 3) Drawing functions ──────────────────────────────────────────────────────
def draw_bbox(im: Image.Image, bbox: list[float], color: str = "red", width: int = 3) -> Image.Image:
    """
    Draws a rectangle given bbox = [x0, y0, x1, y1] on a copy of PIL image.
    """
    canvas = im.copy()
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = bbox
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return canvas

def draw_gaze_path(im: Image.Image, path: list[list[float]], color: str = "lime", width: int = 2) -> Image.Image:
    """
    Given a list of [x,y] coordinates, draws a polyline on top of im.
    If path has fewer than 2 points, returns im unchanged.
    """
    if not path or len(path) < 2:
        return im
    canvas = im.copy()
    draw = ImageDraw.Draw(canvas)
    # Flatten into [(x0,y0), (x1,y1), …]
    pts = [(pt[0], pt[1]) for pt in path]
    draw.line(pts, fill=color, width=width)
    # Optionally draw small circles at each fixation:
    for (x, y) in pts:
        r = 3
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=1)
    return canvas

def draw_rec_point(im: Image.Image, point: list[float], color: str = "yellow", radius: int = 5) -> Image.Image:
    """
    Given a single [x, y], draws a small filled circle.
    """
    if not point or len(point) != 2:
        return im
    x, y = point
    canvas = im.copy()
    draw = ImageDraw.Draw(canvas)
    r = radius
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)
    return canvas

# ─── 4) Main Streamlit UI ─────────────────────────────────────────────────────
example_keys = list(data.keys())
selected_image_path = st.selectbox("🔎 Choose an example by image_path:", example_keys)

entry = data[selected_image_path]
filename = Path(selected_image_path).name
bbox = entry["bbox"]
candidates = entry["candidates"]
gaze_scores = entry["gaze_distances"]
rec_scores = entry["rec_distances"]
gaze_coords = entry["gaze_coords"]
rec_point = entry["rec_point"]

# Load the base image
st.write("### Original Image (no overlays):")
base_image = get_image_by_filename(filename)
if base_image is None:
    st.error(f"Could not find ‘{filename}’ in data/overlayed_images/.")
    st.stop()

# Draw and show the bounding box
st.write("### ① Base image with bounding box:")
img_bbox = draw_bbox(base_image, bbox, color="red", width=3)
st.image(img_bbox, use_column_width=True)

# Let user pick one of the candidate texts
candidate_texts = [cand["text"] for cand in candidates]
sel_idx = st.selectbox("② Pick a candidate text:", list(range(len(candidates))), format_func=lambda i: candidate_texts[i])

sel_candidate = candidates[sel_idx]
sel_text = sel_candidate["text"]
sel_type = sel_candidate["type"]

st.markdown(f"**Selected candidate:** “{sel_text}”  — Type: `{sel_type}`")

# ─── 5) Overlay gaze path and REC point on demand ───────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.write("### ③ Show gaze path for this candidate")
    if gaze_coords:
        img_gaze = draw_gaze_path(base_image, gaze_coords, color="lime", width=2)
        # Also draw the bbox in case you still want to see it:
        img_gaze = draw_bbox(img_gaze, bbox, color="red", width=2)
        st.image(img_gaze, use_column_width=True)
    else:
        st.write("No gaze coordinates available for this candidate.")

with col2:
    st.write("### ④ Show REC point for this candidate")
    if rec_point:
        img_rec = draw_rec_point(base_image, rec_point, color="yellow", radius=5)
        # Also draw the bbox for reference:
        img_rec = draw_bbox(img_rec, bbox, color="red", width=2)
        st.image(img_rec, use_column_width=True)
    else:
        st.write("No REC point available for this candidate.")

# ─── 6) Show the reranked tables side by side ───────────────────────────────────
def reranked_table(candidates_list, distances, title: str):
    sorted_items = sorted(zip(distances, candidates_list), key=lambda x: x[0])
    st.write(f"#### {title}")
    for dist, cand in sorted_items:
        type_tag = "🟡 Gold" if cand["type"] == "gold" else "⚪️ Gen"
        st.markdown(f"- `{dist:.2f}`  **{cand['text']}** ({type_tag})")

st.write("---")
col_a, col_b = st.columns(2)
with col_a:
    reranked_table(candidates, gaze_scores, "👁️ Gaze-Based Ranking")
with col_b:
    reranked_table(candidates, rec_scores, "📍 REC-Based Ranking")

