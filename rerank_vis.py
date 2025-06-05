import streamlit as st
import json
from PIL import Image, ImageDraw
from pathlib import Path

# --- Load JSON files ---
with open("gaze_scoring_results.json") as f:
    gaze_data = json.load(f)

with open("rec_scoring_results.json") as f:
    rec_data = json.load(f)

# --- Index by image_path ---
data = {}
for g_entry, r_entry in zip(gaze_data, rec_data):
    image_path = g_entry["image_path"]
    assert image_path == r_entry["image_path"]
    data[image_path] = {
        "bbox": g_entry["bbox"],
        "candidates": g_entry["candidates"],
        "gaze_distances": g_entry["gaze_distances"],
        "rec_distances": r_entry["rec_distances"]
    }

# --- Load images from local data/ directory ---
def get_image_by_filename(filename):
    local_path = Path("data/overlayed_images/") / filename
    if local_path.exists():
        return Image.open(local_path)
    return None

# --- Streamlit UI ---
st.title("Gaze vs REC Candidate Scoring Viewer")

example_keys = list(data.keys())
selected_key = st.selectbox("Choose an example:", example_keys)

entry = data[selected_key]
filename = Path(selected_key).name
bbox = entry["bbox"]
candidates = entry["candidates"]
gaze_scores = entry["gaze_distances"]
rec_scores = entry["rec_distances"]

# --- Load Image ---
image = get_image_by_filename(filename)
if image is None:
    st.error(f"Image '{filename}' not found in local data/ directory.")
    st.stop()

# # --- Draw BBox on Image ---
# def draw_bbox(image, bbox, color="red", width=4):
#     draw = ImageDraw.Draw(image)
#     draw.rectangle(bbox, outline=color, width=width)
#     return image

st.subheader("Image with Bounding Box")
img_copy = image.copy()
img_with_bbox = draw_bbox(img_copy, bbox)
st.image(img_with_bbox, caption=filename)

# --- Create sortable ranking tables ---
def reranked_table(candidates, distances, title):
    sorted_items = sorted(zip(distances, candidates), key=lambda x: x[0])
    st.subheader(title)
    for dist, cand in sorted_items:
        tag = "🟡 Gold" if cand["type"] == "gold" else "⚪️ Gen"
        st.markdown(f"- `{dist:.2f}`: **{cand['text']}** ({tag})", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    reranked_table(candidates, gaze_scores, "👁️ Gaze-Based Ranking")

with col2:
    reranked_table(candidates, rec_scores, "📍 REC-Based Ranking")
