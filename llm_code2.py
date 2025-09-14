import openai
import cv2
import base64
import os
from io import BytesIO

# --------------- CONFIGURATION ---------------
openai.api_key = "your_openai_api_key"

USE_AZURE = False
LOCAL_IMAGE_PATH = "sample_data/image1.jpg"
bounding_rect = [(50, 60, 100, 120), (200, 150, 80, 90)]
MODEL = "gpt-4-vision-preview"
SIMILARITY_THRESHOLD = 0.85  # Not used now, LLM gives Yes/No


# --------------- HELPERS ----------------

def encode_image_to_base64(cv2_img):
    _, buffer = cv2.imencode('.png', cv2_img)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str


def ask_llm_if_roi_in_image(full_img_b64, roi_b64):
    """
    Use OpenAI's GPT-4 Vision to determine if ROI is present in full image.
    """
    prompt = (
        "You are given two images: the first is a full image, the second is a cropped region (ROI). "
        "Determine if the cropped image is a part of the full image. "
        "Reply only with 'Yes' or 'No'."
    )

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{full_img_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{roi_b64}"}},
            ]}
        ],
        max_tokens=10
    )

    reply = response.choices[0].message.content.strip().lower()
    return 'yes' in reply


# --------------- LOCAL IMAGE LOADER ----------------
def load_image_from_local():
    print(f"Loading image from local path: {LOCAL_IMAGE_PATH}")
    if not os.path.exists(LOCAL_IMAGE_PATH):
        raise FileNotFoundError(f"Image not found at path: {LOCAL_IMAGE_PATH}")
    img = cv2.imread(LOCAL_IMAGE_PATH)
    return img


# --------------- MAIN PROCESS ----------------
def process_image_with_llm(full_img, bounding_boxes):
    full_img_b64 = encode_image_to_base64(full_img)

    for (x, y, w, h) in bounding_boxes:
        crop_roi = full_img[y:y + h, x:x + w]

        if crop_roi.size == 0:
            print(f"Empty ROI at ({x}, {y}, {w}, {h}) - Skipping")
            continue

        roi_b64 = encode_image_to_base64(crop_roi)

        try:
            is_present = ask_llm_if_roi_in_image(full_img_b64, roi_b64)
            print(f"ROI at ({x}, {y}, {w}, {h}) - Present: {is_present}")
        except Exception as e:
            print(f"OpenAI API error for ROI at ({x}, {y}, {w}, {h}): {e}")
            is_present = False

        color = (0, 255, 0) if is_present else (0, 0, 255)
        cv2.rectangle(full_img, (x, y), (x + w, y + h), color, 2)

    return full_img


# --------------- MAIN FUNCTION ----------------
def main():
    if USE_AZURE:
        raise NotImplementedError("Azure loader not included.")
    else:
        img = load_image_from_local()

    result_img = process_image_with_llm(img, bounding_rect)

    # Show or save
    cv2.imshow("LLM Visual Match", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
