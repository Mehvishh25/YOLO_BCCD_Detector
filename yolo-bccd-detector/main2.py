import streamlit as st
import torch
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from ultralytics import YOLO
from collections import defaultdict

# YOLO Model Path
MODEL_PATH = "best.pt"

# Annotations Path
ANNOTATIONS_PATH = "BCCD_Dataset/BCCD/Annotations"

# Load YOLO Model
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Streamlit UI
st.title("YOLOv8 Object Detection with Precision & Recall")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def parse_xml(annotation_path):
    """ Parse XML file to extract ground truth bounding boxes. """
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    gt_boxes = []
    gt_labels = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        x_min = int(bbox.find("xmin").text)
        y_min = int(bbox.find("ymin").text)
        x_max = int(bbox.find("xmax").text)
        y_max = int(bbox.find("ymax").text)

        gt_boxes.append([x_min, y_min, x_max, y_max])
        gt_labels.append(name)

    return gt_boxes, gt_labels

def calculate_iou(box1, box2):
    """ Compute Intersection over Union (IoU) between two bounding boxes. """
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    # Compute intersection area
    inter_x1 = max(x1, x1_gt)
    inter_y1 = max(y1, y1_gt)
    inter_x2 = min(x2, x2_gt)
    inter_y2 = min(y2, y2_gt)

    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    # Compute areas
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)

    # Compute IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Process uploaded image
if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Perform YOLO detection
    results = model.predict(image_np, save=True)
    
    # Extract filename (without extension)
    filename = os.path.splitext(uploaded_file.name)[0]
    
    # Find corresponding annotation XML
    annotation_file = os.path.join(ANNOTATIONS_PATH, f"{filename}.xml")
    
    if os.path.exists(annotation_file):
        gt_boxes, gt_labels = parse_xml(annotation_file)
    else:
        st.error("No matching annotation found for this image!")
        gt_boxes, gt_labels = [], []

    # Process YOLO predictions
    pred_boxes = []
    pred_labels = []
    pred_scores = []

    detected_image = image_np.copy()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for box, label, conf in zip(boxes, labels, confs):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(label)]
            confidence = f"{conf:.2f}"

            pred_boxes.append([x1, y1, x2, y2])
            pred_labels.append(class_name)
            pred_scores.append(conf)

            # Draw bounding boxes on image
            cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(detected_image, f"{class_name} {confidence}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Compute Precision & Recall
    iou_threshold = 0.5  # IoU threshold for TP
    classwise_tp = defaultdict(int)
    classwise_fp = defaultdict(int)
    classwise_fn = defaultdict(int)

    matched_gt = set()

    for i, pred_box in enumerate(pred_boxes):
        pred_label = pred_labels[i]
        max_iou = 0
        matched_gt_idx = -1

        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou and iou > iou_threshold and gt_labels[j] == pred_label:
                max_iou = iou
                matched_gt_idx = j

        if matched_gt_idx != -1 and matched_gt_idx not in matched_gt:
            classwise_tp[pred_label] += 1
            matched_gt.add(matched_gt_idx)
        else:
            classwise_fp[pred_label] += 1

    for j, gt_label in enumerate(gt_labels):
        if j not in matched_gt:
            classwise_fn[gt_label] += 1

    # Compute Precision & Recall for each class
    classwise_precision = {}
    classwise_recall = {}

    for cls in set(gt_labels + pred_labels):
        tp = classwise_tp[cls]
        fp = classwise_fp[cls]
        fn = classwise_fn[cls]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        classwise_precision[cls] = precision
        classwise_recall[cls] = recall

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(detected_image, caption="YOLO Detection", use_container_width=True)

    st.subheader("📊 Precision & Recall Metrics")
    for cls in classwise_precision:
        st.write(f"**{cls}**")
        st.write(f"🔹 Precision: `{classwise_precision[cls]:.2f}`")
        st.write(f"🔹 Recall: `{classwise_recall[cls]:.2f}`")
        st.write("---")
