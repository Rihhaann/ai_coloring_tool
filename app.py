import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Coloring App", layout="wide")

st.title("🎨 AI Coloring from Real Book Photos")
st.write("Upload a **photo or clean drawing outline**. The system will generate cleaned line art and auto-color it.")

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed

# -----------------------------
# REGION SEGMENTATION
# -----------------------------
def get_regions(binary_img):
    h, w = binary_img.shape
    filled = binary_img.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)

    regions = []

    for y in range(h):
        for x in range(w):
            if filled[y, x] == 0:
                flood = filled.copy()
                cv2.floodFill(flood, mask, (x, y), 255)
                region = cv2.subtract(flood, filled)

                area = np.sum(region == 255)
                regions.append((region, area))

                filled = flood

    return regions

# -----------------------------
# COLOR PALETTE
# -----------------------------
PALETTE = [
    (255, 0, 0),     # Red
    (255, 255, 0),   # Yellow
    (255, 165, 0),   # Orange
    (138, 43, 226),  # Violet
    (0, 255, 0),     # Green
    (165, 42, 42),   # Brown
    (0, 0, 0)        # Black
]

# -----------------------------
# COLORIZATION
# -----------------------------
def colorize(binary, original):
    colored = np.ones_like(original) * 255
    regions = get_regions(binary)

    # Identify background (largest region)
    largest_area = max(regions, key=lambda x: x[1])[1]

    color_index = 0
    for region, area in regions:
        if area == largest_area:
            continue  # 🚫 Skip background

        mask = region == 255
        color = PALETTE[color_index % len(PALETTE)]
        colored[mask] = color
        color_index += 1

    # Preserve outlines
    edges = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(colored, colored, mask=cv2.bitwise_not(binary))
    result = cv2.add(result, edges)

    return result

# -----------------------------
# STREAMLIT UI
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload drawing image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Original Image
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Preprocessing
    binary = preprocess_image(image_np)

    # Cleaned Line Art
    st.subheader("Cleaned Line Art (AI Vision)")
    st.image(binary, clamp=True, use_container_width=True)

    # ✅ Download Cleaned Line Art
    lineart_pil = Image.fromarray(binary)
    buf_lineart = io.BytesIO()
    lineart_pil.save(buf_lineart, format="PNG")
    lineart_bytes = buf_lineart.getvalue()

    st.download_button(
        label="⬇️ Download Cleaned Line Art",
        data=lineart_bytes,
        file_name="cleaned_line_art.png",
        mime="image/png"
    )

    # Colorization
    result = colorize(binary, image_np)

    st.subheader("🎨 Auto Colored Result")
    st.image(result, use_container_width=True)

    # ✅ Download Colored Image
    result_pil = Image.fromarray(result)
    buf_color = io.BytesIO()
    result_pil.save(buf_color, format="PNG")
    color_bytes = buf_color.getvalue()

    st.download_button(
        label="⬇️ Download Colored Image",
        data=color_bytes,
        file_name="colored_output.png",
        mime="image/png"
    )







