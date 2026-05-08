from datetime import datetime
import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from src import GRFBUNet


# USER SETTINGS - CHANGE THESE ONLY

INPUT_PATH = '/Users/mohammed-alsulami/Documents/GitHub/CV-Accessibility-of-Public-Transport/Sprint-3 Model/Test Model img-vids/test1.mp4'

MODEL_PATH = "/Users/mohammed-alsulami/Documents/GitHub/CV-Accessibility-of-Public-Transport/Sprint-3 Model/Model/model.pth"

TEMPLATE_PATH = '/Users/mohammed-alsulami/Documents/GitHub/CV-Accessibility-of-Public-Transport/Sprint-3 Model/Report_Template.pdf'

OUTPUT_PDF_PATH = "/Users/mohammed-alsulami/Documents/GitHub/CV-Accessibility-of-Public-Transport/Sprint-3 Model/output_report.pdf"

THRESHOLD = 500

FRAME_INTERVAL_SECONDS = 1
MAX_VIDEO_SECONDS = 10

# DSAPT contrast levels
DSAPT_MIN_CONTRAST = 30
DSAPT_MEDIUM_CONTRAST = 45
DSAPT_HIGH_CONTRAST = 60

# PDF text position for accessibility feature
# If the text appears in the wrong place, only adjust this Y value.
ACCESSIBILITY_FEATURE_X = 205.33
ACCESSIBILITY_FEATURE_Y = 345


# Load Model
def load_model(model_path, device):
    classes = 1
    model = GRFBUNet(in_channels=3, num_classes=classes + 1, base_c=32)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model


# Preprocess Image
def preprocess_image(image):
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    transform = transforms.Compose([
        transforms.Resize(565),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform(image)


# Create Overlay
def create_overlay(original_img, prediction_mask):
    orig_np = np.array(original_img).copy()
    mask_np = np.array(prediction_mask)

    binary_mask = mask_np > 0

    overlay = orig_np.copy()
    overlay[binary_mask] = [0, 255, 0]

    blended = (0.6 * orig_np + 0.4 * overlay).astype(np.uint8)

    return blended


# Check Frame Quality
def get_frame_quality_score(frame):
    """
    Higher score means better frame.
    This checks:
    1. Brightness
    2. Sharpness / blur
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Reject very dark frames
    if brightness < 50:
        return 0

    # Reject very blurry frames
    if blur_score < 100:
        return 0

    # Combined quality score
    quality_score = brightness + blur_score

    return quality_score


# Run Model on One Image
def run_model_on_image(model, original_img, device, threshold=500):
    original_w, original_h = original_img.size

    img = preprocess_image(original_img)
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        start_time = time.time()
        output = model(img.to(device))
        end_time = time.time()

        prediction = output["out"].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    prediction = Image.fromarray(prediction)
    prediction = prediction.resize((original_w, original_h), resample=Image.NEAREST)
    prediction = np.array(prediction)

    prediction[prediction == 1] = 255
    prediction[prediction == 0] = 0

    mask_img = Image.fromarray(prediction).convert("L")
    overlay_img = create_overlay(original_img, mask_img)

    detected_pixels = int(np.sum(np.array(mask_img) > 0))
    result = "Yes" if detected_pixels > threshold else "No"

    inference_time = end_time - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0.0

    return result, detected_pixels, inference_time, fps, overlay_img, mask_img


# Calculate Luminance Contrast
# Calculate Luminance Contrast
def calculate_luminance_contrast(original_img, mask_img):
    """
    Calculates approximate luminance contrast between:
    1. Raised tactile indicators inside the detected tactile region
    2. Surrounding non-tactile floor area

    It checks both dark and light tactile indicators, then uses the one
    with the stronger contrast.
    """

    img_np = np.array(original_img).astype(np.float32)
    mask_np = np.array(mask_img)

    tactile_mask = mask_np > 0
    surrounding_mask = mask_np == 0

    if np.sum(tactile_mask) == 0:
        return 0.0, 0.0, 0.0

    if np.sum(surrounding_mask) == 0:
        return 0.0, 0.0, 0.0

    r = img_np[:, :, 0]
    g = img_np[:, :, 1]
    b = img_np[:, :, 2]

    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    tactile_luminance_values = luminance[tactile_mask]

    surrounding_luminance = np.mean(luminance[surrounding_mask])

    # Dark raised indicators candidate
    dark_threshold = np.percentile(tactile_luminance_values, 25)
    dark_tactile_mask = tactile_mask & (luminance <= dark_threshold)

    # Light raised indicators candidate
    light_threshold = np.percentile(tactile_luminance_values, 75)
    light_tactile_mask = tactile_mask & (luminance >= light_threshold)

    dark_luminance = np.mean(luminance[dark_tactile_mask]) if np.sum(dark_tactile_mask) > 0 else 0.0
    light_luminance = np.mean(luminance[light_tactile_mask]) if np.sum(light_tactile_mask) > 0 else 0.0

    def contrast(lum1, lum2):
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)

        if lighter == 0:
            return 0.0

        return ((lighter - darker) / lighter) * 100

    dark_contrast = contrast(dark_luminance, surrounding_luminance)
    light_contrast = contrast(light_luminance, surrounding_luminance)

    if dark_contrast >= light_contrast:
        contrast_percentage = dark_contrast
        tactile_luminance = dark_luminance
    else:
        contrast_percentage = light_contrast
        tactile_luminance = light_luminance

    return contrast_percentage, tactile_luminance, surrounding_luminance


# DSAPT Contrast Compatibility
def get_dsapt_compatibility(result, contrast_percentage):
    """
    Converts luminance contrast into DSAPT compatibility score.

    Score logic:
    No tactile flooring detected = 0%
    Contrast < 30% = 0%
    Contrast 30% to 44.99% = 50%
    Contrast 45% to 59.99% = 75%
    Contrast >= 60% = 100%
    """

    if result == "No":
        compatibility_score = 0
        compatibility_label = "Not assessed"
        notes = (
            "Tactile flooring was not detected in the input image, so DSAPT "
            "luminance contrast compatibility could not be assessed."
        )

    elif contrast_percentage < DSAPT_MIN_CONTRAST:
        compatibility_score = 0
        compatibility_label = "Not compatible"
        notes = (
            f"Tactile flooring was detected, but the estimated luminance contrast "
            f"is {contrast_percentage:.2f}%. This is below the minimum selected "
            f"DSAPT contrast level of {DSAPT_MIN_CONTRAST}%, so it is not considered "
            f"compatible based on contrast."
        )

    elif contrast_percentage < DSAPT_MEDIUM_CONTRAST:
        compatibility_score = 50
        compatibility_label = "Minimum compatibility"
        notes = (
            f"Tactile flooring was detected with an estimated luminance contrast "
            f"of {contrast_percentage:.2f}%. This meets the minimum selected DSAPT "
            f"contrast level of {DSAPT_MIN_CONTRAST}%, but does not reach the "
            f"{DSAPT_MEDIUM_CONTRAST}% or {DSAPT_HIGH_CONTRAST}% levels. Therefore, "
            f"it is considered partially compatible based on contrast."
        )

    elif contrast_percentage < DSAPT_HIGH_CONTRAST:
        compatibility_score = 75
        compatibility_label = "Moderate compatibility"
        notes = (
            f"Tactile flooring was detected with an estimated luminance contrast "
            f"of {contrast_percentage:.2f}%. This exceeds the {DSAPT_MEDIUM_CONTRAST}% "
            f"contrast level but does not reach the {DSAPT_HIGH_CONTRAST}% high-contrast "
            f"level. Therefore, it shows moderate DSAPT contrast compatibility."
        )

    else:
        compatibility_score = 100
        compatibility_label = "High compatibility"
        notes = (
            f"Tactile flooring was detected with an estimated luminance contrast "
            f"of {contrast_percentage:.2f}%. This exceeds the {DSAPT_HIGH_CONTRAST}% "
            f"contrast level, so it is considered highly compatible based on the "
            f"selected DSAPT contrast criteria."
        )

    return compatibility_score, compatibility_label, notes


# PDF Report Generation
def pdf_report(input_image, processed_image, accessibility_feature,
               dsapt_compliance_score, notes,
               output_path="output_report.pdf", template_path="Report_Template.pdf"):

    import io
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from pdfrw import PdfReader, PdfWriter, PageMerge

    PAGE_W, PAGE_H = 612, 792

    # Convert input to PIL Image
    # Accepts file path, PIL Image, or numpy array
    def to_pil(img_input):
        if isinstance(img_input, np.ndarray):
            return Image.fromarray(img_input.astype(np.uint8)).convert("RGB")

        if isinstance(img_input, Image.Image):
            return img_input.convert("RGB")

        if isinstance(img_input, str) and os.path.exists(img_input):
            return Image.open(img_input).convert("RGB")

        return None

    def draw_image_in_cell(c, pil_img, cell_x, cell_y_bottom, cell_w, cell_h, padding=4):
        iw, ih = pil_img.size

        max_w = cell_w - 2 * padding
        max_h = cell_h - 2 * padding

        scale = min(max_w / iw, max_h / ih)

        draw_w = iw * scale
        draw_h = ih * scale

        draw_x = cell_x + (cell_w - draw_w) / 2
        draw_y = cell_y_bottom + (cell_h - draw_h) / 2

        c.drawImage(ImageReader(pil_img), draw_x, draw_y, draw_w, draw_h)

    def draw_wrapped_text(c, text, x, y, max_width, font_name, font_size, line_height):
        words = str(text).split()
        lines = []
        current = ""

        for word in words:
            test = (current + " " + word).strip()

            if c.stringWidth(test, font_name, font_size) <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        for line in lines:
            c.drawString(x, y, line)
            y -= line_height

    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=(PAGE_W, PAGE_H))

    # 1. Date
    c.setFont("Helvetica", 11)
    c.drawString(205.33, 465.15, datetime.now().strftime("%d/%m/%Y"))

    # 2. Input image
    pil_in = to_pil(input_image)
    if pil_in:
        draw_image_in_cell(c, pil_in, 205.33, 360, 540 - 205.33, 100)

    # 3. Accessibility feature detected
    c.setFont("Helvetica", 11)
    c.drawString(ACCESSIBILITY_FEATURE_X, ACCESSIBILITY_FEATURE_Y, accessibility_feature)

    # 4. Processed/output image
    pil_out = to_pil(processed_image)
    if pil_out:
        draw_image_in_cell(c, pil_out, 205.33, 226, 540 - 205.33, 100)

    # 5. DSAPT compatibility score
    c.setFont("Helvetica", 11)
    c.drawString(205.33, 210.58, f"{dsapt_compliance_score}%")

    # 6. Notes
    draw_wrapped_text(c, notes, 205.33, 185.55, 290, "Helvetica", 10, 12)

    c.save()
    packet.seek(0)

    template = PdfReader(template_path)
    overlay_pdf = PdfReader(packet)

    PageMerge(template.pages[0]).add(overlay_pdf.pages[0]).render()
    PdfWriter(output_path, trailer=template).write()

    return output_path


# Process Image Input
def process_image(image_path, model, device, threshold=500):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_img = Image.open(image_path).convert("RGB")

    result, detected_pixels, inference_time, fps, overlay_img, mask_img = run_model_on_image(
        model=model,
        original_img=original_img,
        device=device,
        threshold=threshold
    )

    print("Input type: Image")
    print(f"Detected pixels: {detected_pixels}")
    print(f"Tactile flooring detected: {result}")

    

    return original_img, overlay_img, mask_img, result, detected_pixels, inference_time, fps


# Process Video Input

def process_video(video_path, model, device, threshold=500,
                  frame_interval_seconds=1, max_video_seconds=10):

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps if video_fps > 0 else 0

    print("Input type: Video")
    print(f"Video duration: {video_duration:.2f} seconds")

    if video_duration > max_video_seconds:
        print(f"Warning: The video is longer than {max_video_seconds} seconds.")
        print("Only selected frames from the video will be checked.")

    frame_step = int(video_fps * frame_interval_seconds)

    if frame_step <= 0:
        frame_step = 1

    selected_frames = []
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Only check frames at selected intervals
        if frame_number % frame_step == 0:
            quality_score = get_frame_quality_score(frame)

            if quality_score > 0:
                selected_frames.append({
                    "frame_number": frame_number,
                    "frame": frame.copy(),
                    "quality_score": quality_score
                })

        frame_number += 1

    cap.release()

    if len(selected_frames) == 0:
        print("No good-quality frames were found.")
        return None, None, None, None, None, None, None

    # Choose the best frame based on brightness and sharpness
    best_frame_info = max(selected_frames, key=lambda x: x["quality_score"])
    best_frame = best_frame_info["frame"]
    best_frame_number = best_frame_info["frame_number"]

    # Convert OpenCV BGR frame to RGB PIL image
    best_frame_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    best_pil_img = Image.fromarray(best_frame_rgb)

    result, detected_pixels, inference_time, fps, overlay_img, mask_img = run_model_on_image(
        model=model,
        original_img=best_pil_img,
        device=device,
        threshold=threshold
    )

    print(f"Checked frames: {len(selected_frames)}")
    print(f"Best frame number: {best_frame_number}")
    print(f"Best frame quality score: {best_frame_info['quality_score']:.2f}")
    print(f"Detected pixels: {detected_pixels}")
    print(f"Tactile flooring detected: {result}")

    

    return best_pil_img, overlay_img, mask_img, result, detected_pixels, inference_time, fps


# Detect Input Type
def is_image_file(file_path):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    ext = os.path.splitext(file_path)[1].lower()

    return ext in image_extensions


def is_video_file(file_path):
    video_extensions = [".mp4", ".mov", ".avi", ".mkv"]
    ext = os.path.splitext(file_path)[1].lower()

    return ext in video_extensions


# Main Function
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"PDF template not found: {TEMPLATE_PATH}")

    model = load_model(MODEL_PATH, device)

    if is_image_file(INPUT_PATH):
        original_img, overlay_img, mask_img, result, detected_pixels, inference_time, fps = process_image(
            image_path=INPUT_PATH,
            model=model,
            device=device,
            threshold=THRESHOLD
        )

    elif is_video_file(INPUT_PATH):
        original_img, overlay_img, mask_img, result, detected_pixels, inference_time, fps = process_video(
            video_path=INPUT_PATH,
            model=model,
            device=device,
            threshold=THRESHOLD,
            frame_interval_seconds=FRAME_INTERVAL_SECONDS,
            max_video_seconds=MAX_VIDEO_SECONDS
        )

        if original_img is None:
            print("PDF report was not generated because no good-quality frame was found.")
            return

    else:
        raise ValueError("Unsupported file type. Please use an image or video file.")

    # Accessibility Feature
    if result == "Yes":
        accessibility_feature = "Accessibility Feature Detected: Tactile flooring"
    else:
        accessibility_feature = "No Accessibility Feature Detected"

    # DSAPT Contrast-Based Report Values
    contrast_percentage, tactile_luminance, surrounding_luminance = calculate_luminance_contrast(
    original_img, mask_img)

    dsapt_compliance_score, dsapt_compatibility_label, notes = get_dsapt_compatibility(
        result=result,
        contrast_percentage=contrast_percentage
    )

    
    print(f"Tactile area luminance: {tactile_luminance:.2f}")
    print(f"Surrounding area luminance: {surrounding_luminance:.2f}")
    print(f"Estimated luminance contrast: {contrast_percentage:.2f}%")
    print(f"DSAPT compatibility score: {dsapt_compliance_score}%")
    print(f"DSAPT compatibility label: {dsapt_compatibility_label}")
    

    # Generate PDF Report
    report_path = pdf_report(
        input_image=original_img,
        processed_image=overlay_img,
        accessibility_feature=accessibility_feature,
        dsapt_compliance_score=dsapt_compliance_score,
        notes=notes,
        output_path=OUTPUT_PDF_PATH,
        template_path=TEMPLATE_PATH
    )

    print(f"PDF report generated successfully: {report_path}")


if __name__ == "__main__":
    main()
