from datetime import datetime
import os
import time
import uuid

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src import GRFBUNet


# PATH SETTINGS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pth")
TEMPLATE_PATH = os.path.join(BASE_DIR, "report_template.pdf")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

THRESHOLD = 500

FRAME_INTERVAL_SECONDS = 1
MAX_VIDEO_SECONDS = 10


# DSAPT CONTRAST LEVELS

DSAPT_MIN_CONTRAST = 30
DSAPT_MEDIUM_CONTRAST = 45
DSAPT_HIGH_CONTRAST = 60


# PDF TEXT POSITION

ACCESSIBILITY_FEATURE_X = 205.33
ACCESSIBILITY_FEATURE_Y = 345


# LOAD MODEL

def load_model(model_path, device):
    classes = 1
    model = GRFBUNet(in_channels=3, num_classes=classes + 1, base_c=32)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    model.to(device)
    model.eval()

    return model


# PREPROCESS IMAGE

def preprocess_image(image):
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    transform = transforms.Compose([
        transforms.Resize(565),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform(image)


# CREATE OVERLAY

def create_overlay(original_img, prediction_mask):
    orig_np = np.array(original_img).copy()
    mask_np = np.array(prediction_mask)

    binary_mask = mask_np > 0

    overlay = orig_np.copy()
    overlay[binary_mask] = [0, 255, 0]

    blended = (0.6 * orig_np + 0.4 * overlay).astype(np.uint8)

    return blended


# FRAME QUALITY CHECK

def get_frame_quality_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if brightness < 50:
        return 0

    if blur_score < 100:
        return 0

    quality_score = brightness + blur_score

    return quality_score


# RUN MODEL ON ONE IMAGE

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


# CALCULATE LUMINANCE CONTRAST

def calculate_luminance_contrast(original_img, mask_img):
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

    dark_threshold = np.percentile(tactile_luminance_values, 25)
    dark_tactile_mask = tactile_mask & (luminance <= dark_threshold)

    light_threshold = np.percentile(tactile_luminance_values, 75)
    light_tactile_mask = tactile_mask & (luminance >= light_threshold)

    dark_luminance = (
        np.mean(luminance[dark_tactile_mask])
        if np.sum(dark_tactile_mask) > 0
        else 0.0
    )

    light_luminance = (
        np.mean(luminance[light_tactile_mask])
        if np.sum(light_tactile_mask) > 0
        else 0.0
    )

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


# DSAPT COMPATIBILITY

def get_dsapt_compatibility(result, contrast_percentage):
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


# PDF REPORT GENERATION

def pdf_report(
    input_image,
    processed_image,
    accessibility_feature,
    dsapt_compliance_score,
    notes,
    output_path,
    template_path
):
    import io
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from pdfrw import PdfReader, PdfWriter, PageMerge

    PAGE_W, PAGE_H = 612, 792

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

    c.setFont("Helvetica", 11)
    c.drawString(205.33, 465.15, datetime.now().strftime("%d/%m/%Y"))

    pil_in = to_pil(input_image)
    if pil_in:
        draw_image_in_cell(c, pil_in, 205.33, 360, 540 - 205.33, 100)

    c.setFont("Helvetica", 11)
    c.drawString(ACCESSIBILITY_FEATURE_X, ACCESSIBILITY_FEATURE_Y, accessibility_feature)

    pil_out = to_pil(processed_image)
    if pil_out:
        draw_image_in_cell(c, pil_out, 205.33, 226, 540 - 205.33, 100)

    c.setFont("Helvetica", 11)
    c.drawString(205.33, 210.58, f"{dsapt_compliance_score}%")

    c.setFont("Helvetica", 10)
    draw_wrapped_text(c, notes, 205.33, 185.55, 290, "Helvetica", 10, 12)

    c.save()
    packet.seek(0)

    template = PdfReader(template_path)
    overlay_pdf = PdfReader(packet)

    PageMerge(template.pages[0]).add(overlay_pdf.pages[0]).render()
    PdfWriter(output_path, trailer=template).write()

    return output_path


# PROCESS IMAGE

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

    return original_img, overlay_img, mask_img, result, detected_pixels, inference_time, fps


# PROCESS VIDEO

def process_video(
    video_path,
    model,
    device,
    threshold=500,
    frame_interval_seconds=1,
    max_video_seconds=10
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps if video_fps > 0 else 0

    frame_step = int(video_fps * frame_interval_seconds)

    if frame_step <= 0:
        frame_step = 1

    selected_frames = []
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

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
        raise ValueError("No good-quality frames were found in the video.")

    best_frame_info = max(selected_frames, key=lambda x: x["quality_score"])
    best_frame = best_frame_info["frame"]
    best_frame_number = best_frame_info["frame_number"]

    best_frame_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    best_pil_img = Image.fromarray(best_frame_rgb)

    result, detected_pixels, inference_time, fps, overlay_img, mask_img = run_model_on_image(
        model=model,
        original_img=best_pil_img,
        device=device,
        threshold=threshold
    )

    video_info = {
        "video_duration": round(float(video_duration), 2),
        "checked_frames": len(selected_frames),
        "best_frame_number": int(best_frame_number),
        "best_frame_quality_score": round(float(best_frame_info["quality_score"]), 2),
        "video_warning": video_duration > max_video_seconds
    }

    return best_pil_img, overlay_img, mask_img, result, detected_pixels, inference_time, fps, video_info


# FILE TYPE CHECK

def is_image_file(file_path):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    ext = os.path.splitext(file_path)[1].lower()

    return ext in image_extensions


def is_video_file(file_path):
    video_extensions = [".mp4", ".mov", ".avi", ".mkv"]
    ext = os.path.splitext(file_path)[1].lower()

    return ext in video_extensions


# MAIN FUNCTION FOR BACKEND

def run_model(input_path):
    """
    This is the function that main.py should call.

    Example from main.py:
        result = run_model(file_path)

    input_path comes from the uploaded file.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"PDF template not found: {TEMPLATE_PATH}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, device)

    video_info = None

    if is_image_file(input_path):
        input_type = "Image"

        original_img, overlay_img, mask_img, result, detected_pixels, inference_time, fps = process_image(
            image_path=input_path,
            model=model,
            device=device,
            threshold=THRESHOLD
        )

    elif is_video_file(input_path):
        input_type = "Video"

        original_img, overlay_img, mask_img, result, detected_pixels, inference_time, fps, video_info = process_video(
            video_path=input_path,
            model=model,
            device=device,
            threshold=THRESHOLD,
            frame_interval_seconds=FRAME_INTERVAL_SECONDS,
            max_video_seconds=MAX_VIDEO_SECONDS
        )

    else:
        raise ValueError("Unsupported file type. Please upload an image or video file.")

    if result == "Yes":
        accessibility_feature = "Accessibility Feature Detected: Tactile flooring"
    else:
        accessibility_feature = "No Accessibility Feature Detected"

    contrast_percentage, tactile_luminance, surrounding_luminance = calculate_luminance_contrast(
        original_img,
        mask_img
    )

    dsapt_compliance_score, dsapt_compatibility_label, notes = get_dsapt_compatibility(
        result=result,
        contrast_percentage=contrast_percentage
    )

    report_filename = f"analysis_report_{uuid.uuid4().hex}.pdf"
    report_path = os.path.join(OUTPUT_FOLDER, report_filename)

    pdf_report(
        input_image=original_img,
        processed_image=overlay_img,
        accessibility_feature=accessibility_feature,
        dsapt_compliance_score=dsapt_compliance_score,
        notes=notes,
        output_path=report_path,
        template_path=TEMPLATE_PATH
    )

    response = {
        "message": "Analysis completed successfully",
        "input_type": input_type,
        "tactile_detected": result,
        "detected_pixels": int(detected_pixels),
        "inference_time_seconds": round(float(inference_time), 4),
        "fps": round(float(fps), 2),
        "contrast_percentage": round(float(contrast_percentage), 2),
        "tactile_luminance": round(float(tactile_luminance), 2),
        "surrounding_luminance": round(float(surrounding_luminance), 2),
        "dsapt_compliance_score": int(dsapt_compliance_score),
        "dsapt_compatibility_label": dsapt_compatibility_label,
        "notes": notes,
        "report_filename": report_filename,
        "report_path": report_path
    }

    if video_info is not None:
        response["video_info"] = video_info

    return response


# OPTIONAL LOCAL TEST

if __name__ == "__main__":
    test_input = "test1.jpg"

    result = run_model(test_input)

    print(result)
