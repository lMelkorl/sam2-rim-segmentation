#!/usr/bin/env python3
"""
Web Test Application for Trained Rim Model
- Full car images: Step1 (wheel detection) + Step2 (trained model on crops)
- Crop images: Direct trained model
"""

from flask import Flask, request, render_template, jsonify, send_file
import os
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from PIL import Image
import io
import json
from datetime import datetime
import torch
import sys

# Add SAM2 to path (from gr.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_PKG = os.path.join(BASE_DIR, "sam2")
sys.path.insert(0, SAM2_PKG)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'web_uploads'
app.config['RESULTS_FOLDER'] = 'web_results'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Device selection (from gr.py)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"🖥️ Device: {device}")

# Load trained crop model
MODEL_PATH = "runs/segment/rim_crop_segmentation_v8m/weights/best.pt"
crop_model = None

try:
    crop_model = YOLO(MODEL_PATH)
    print(f"✅ Crop model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Crop model load failed: {e}")

# Load SAM2 for wheel detection (from gr.py)
sam2 = None
mask_generator = None

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    
    CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "sam2.1_hiera_large.pt")
    CONFIG_CANDIDATES = [
        os.path.join(SAM2_PKG, "configs", "sam2.1", "sam2.1_hiera_l.yaml"),
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_l.yaml",
    ]
    
    for cfg in CONFIG_CANDIDATES:
        try:
            sam2 = build_sam2(cfg, CHECKPOINT, device=device, apply_postprocessing=False)
            print(f"✅ SAM2 loaded: {cfg}")
            break
        except Exception as e:
            print(f"❌ SAM2 failed {cfg}: {e}")
    
    if sam2:
        mask_generator = SAM2AutomaticMaskGenerator(sam2)
        print("✅ SAM2 mask generator ready")
        
except Exception as e:
    print(f"❌ SAM2 setup failed: {e}")

class RimDetector:
    def __init__(self, crop_model, mask_generator=None):
        self.crop_model = crop_model
        self.mask_generator = mask_generator
    
    def load_image(self, path, max_dim=1024):
        """Load and resize image (from gr.py)"""
        img = Image.open(path)
        w, h = img.size
        scale = min(max_dim/max(w, h), 1.0)
        img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
        return np.array(img.convert("RGB"))
    
    def filter_rims(self, masks, min_area=1500, circ_thresh=0.65):
        """Filter potential rim regions (from gr.py)"""
        rims = []
        for ann in masks:
            seg = ann["segmentation"].astype(np.uint8)
            area = ann["area"]
            if area < min_area: 
                continue

            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: 
                continue
            cnt = max(cnts, key=cv2.contourArea)

            peri = cv2.arcLength(cnt, True)
            circ = 4*np.pi*area/(peri*peri+1e-6)
            if circ < circ_thresh: 
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ar = w/h
            if not (0.8 <= ar <= 1.2): 
                continue

            rims.append((circ, area, (x, y, x+w, y+h)))

        if len(rims) < 2:
            return [box for _, _, box in rims]

        # Find best pair of rims
        candidates = []
        for i in range(len(rims)):
            for j in range(i+1, len(rims)):
                _, a1, b1 = rims[i]
                _, a2, b2 = rims[j]
                x1, y1, x2, y2 = b1
                x3, y3, x4, y4 = b2
                dx = abs((x1+x2)/2 - (x3+x4)/2)
                dy = abs((y1+y2)/2 - (y3+y4)/2)
                size_sim = min(a1, a2)/max(a1, a2)
                if size_sim > 0.6 and dx > dy:
                    score = dx * size_sim
                    candidates.append((score, b1, b2))
        
        if candidates:
            best = max(candidates, key=lambda x: x[0])
            return [best[1], best[2]]

        return [box for _, _, box in sorted(rims, key=lambda x: -x[0])[:2]]
    
    def detect_wheels_in_full_image(self, image_path):
        """Step 1: Detect wheel regions in full car image"""
        if not self.mask_generator:
            return []
        
        try:
            image = self.load_image(image_path)
            masks = self.mask_generator.generate(image)
            boxes = self.filter_rims(masks)
            
            wheel_regions = []
            for i, (x0, y0, x1, y1) in enumerate(boxes):
                wheel_regions.append({
                    'id': int(i),
                    'box': [int(x0), int(y0), int(x1), int(y1)],
                    'crop': image[y0:y1, x0:x1]
                })
            
            return wheel_regions, image
            
        except Exception as e:
            print(f"❌ Wheel detection error: {e}")
            return [], None
    
    def detect_rim_in_crop(self, crop_image, confidence_threshold=0.3):
        """Step 2: Detect rim in crop using trained model"""
        if not self.crop_model:
            return []
        
        try:
            # Save crop temporarily
            temp_path = "temp_crop.jpg"
            Image.fromarray(crop_image).save(temp_path)
            
            # Run trained model
            results = self.crop_model(temp_path, verbose=False)
            
            detections = []
            for r in results:
                if r.masks is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confidences = r.boxes.conf.cpu().numpy()
                    masks = r.masks.data.cpu().numpy()
                    
                    for box, conf, mask in zip(boxes, confidences, masks):
                        # Apply confidence threshold
                        if conf < confidence_threshold:
                            continue
                            
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Resize mask to crop size
                        mask_resized = cv2.resize(mask, (crop_image.shape[1], crop_image.shape[0]))
                        
                        detection = {
                            'box': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'mask': mask_resized,
                            'area': int(np.sum(mask_resized > 0.5))
                        }
                        detections.append(detection)
            
            # Sort by confidence and take only the best one
            if detections:
                detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
                detections = [detections[0]]  # Keep only the highest confidence detection
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return detections
            
        except Exception as e:
            print(f"❌ Rim detection error: {e}")
            return []
    
    def detect_rims_full_pipeline(self, image_path):
        """Full pipeline: Step1 + Step2"""
        print(f"🔍 Full pipeline: {image_path}")
        
        # Step 1: Detect wheels
        wheel_regions, full_image = self.detect_wheels_in_full_image(image_path)
        print(f"🎯 Step 1: {len(wheel_regions)} wheel regions detected")
        
        if not wheel_regions:
            return [], full_image
        
        # Step 2: Detect rims in each wheel region
        all_detections = []
        
        for wheel_region in wheel_regions:
            crop = wheel_region['crop']
            wheel_box = wheel_region['box']
            
            rim_detections = self.detect_rim_in_crop(crop)
            
            # Map rim detections back to full image coordinates
            for rim_det in rim_detections:
                x1, y1, x2, y2 = rim_det['box']
                
                # Map to full image coordinates
                full_x1 = wheel_box[0] + x1
                full_y1 = wheel_box[1] + y1
                full_x2 = wheel_box[0] + x2
                full_y2 = wheel_box[1] + y2
                
                # Map mask to full image
                full_mask = np.zeros((full_image.shape[0], full_image.shape[1]), dtype=np.uint8)
                crop_h, crop_w = crop.shape[:2]
                mask_resized = cv2.resize(rim_det['mask'], (crop_w, crop_h))
                
                wheel_x0, wheel_y0, wheel_x1, wheel_y1 = wheel_box
                full_mask[wheel_y0:wheel_y1, wheel_x0:wheel_x1] = (mask_resized > 0.5).astype(np.uint8) * 255
                
                full_detection = {
                    'wheel_region_id': int(wheel_region['id']),
                    'box': [int(full_x1), int(full_y1), int(full_x2), int(full_y2)],
                    'confidence': float(rim_det['confidence']),
                    'mask': full_mask,
                    'area': int(rim_det['area']),
                    'center': [int((full_x1 + full_x2) / 2), int((full_y1 + full_y2) / 2)],
                    'wheel_box': [int(x) for x in wheel_box]
                }
                all_detections.append(full_detection)
        
        print(f"🎯 Step 2: {len(all_detections)} rims detected")
        return all_detections, full_image
    
    def detect_rim_crop_only(self, image_path):
        """Direct crop detection (for crop images)"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        detections = self.detect_rim_in_crop(image_rgb)
        return detections, image_rgb
    


    def visualize_results(self, image, detections, is_full_image=True, replacement_mode=False, replacement_image=None):
        """Visualize detection results"""
        result_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            confidence = det['confidence']
            mask = det['mask']
            
            # Only draw bounding boxes if NOT in replacement mode
            if not replacement_mode:
                # Draw bounding box
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw confidence
                label = f"Rim {confidence:.2f}"
                cv2.putText(result_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw wheel region (if full image)
                if is_full_image and 'wheel_box' in det:
                    wx1, wy1, wx2, wy2 = det['wheel_box']
                    cv2.rectangle(result_image, (wx1, wy1), (wx2, wy2), (255, 255, 0), 2)
            
            # Handle rim replacement mode
            if replacement_mode and replacement_image is not None and mask.sum() > 0:
                result_image = self.replace_rim_in_mask(result_image, mask, replacement_image)
            elif mask.sum() > 0:
                # Normal mask overlay visualization
                colored_mask = np.zeros_like(result_image)
                colored_mask[mask > 0] = [255, 0, 0]  # Red
                result_image = cv2.addWeighted(result_image, 0.75, colored_mask, 0.25, 0)
        
        return result_image
    


    def replace_rim_in_mask(self, image, mask, replacement_image):
        """
        Overlay replacement_image onto image using mask, preserving resolution.
        """
        mask_bin = (mask > 0).astype(np.uint8)
        coords = np.column_stack(np.where(mask_bin))
        if coords.size == 0:
            return image  # maske yoksa orijinali döndür

        # Maske bölgesinin bounding box'u
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        roi_h = y_max - y_min + 1
        roi_w = x_max - x_min + 1

        img_h, img_w = image.shape[:2]
        rep_h, rep_w = replacement_image.shape[:2]

        # 1) Eğer replacement image tam olarak full-image boyundaysa, direk piksel eşleştir
        if (rep_h, rep_w) == (img_h, img_w):
            result = image.copy()
            for c in range(3):
                channel = result[..., c]
                channel[mask_bin == 1] = replacement_image[..., c][mask_bin == 1]
            return result

        # 2) Aksi durumda, sadece ROI boyutuna ölçekle (up/down) ve mask bölgesine uygula
        #    - Küçültme için INTER_AREA, büyütme için INTER_CUBIC kullanıyoruz
        interp = cv2.INTER_AREA if (rep_h > roi_h or rep_w > roi_w) else cv2.INTER_CUBIC
        rep_resized = cv2.resize(replacement_image, (roi_w, roi_h), interpolation=interp)

        result = image.copy()
        mask_roi = mask_bin[y_min:y_max+1, x_min:x_max+1]
        for c in range(3):
            roi_channel = result[y_min:y_max+1, x_min:x_max+1, c]
            roi_channel[mask_roi == 1] = rep_resized[..., c][mask_roi == 1]
            result[y_min:y_max+1, x_min:x_max+1, c] = roi_channel

        return result


    def process_batch_images(self, image_paths, detection_type='auto'):
        """Process multiple images in batch"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"🔍 Processing batch image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                # Detect based on type
                if detection_type == 'crop':
                    detections, image = self.detect_rim_crop_only(image_path)
                    is_full_image = False
                else:
                    detections, image = self.detect_rims_full_pipeline(image_path)
                    is_full_image = True
                
                # Create result visualization
                result_image = self.visualize_results(image, detections, is_full_image)
                
                # Save result
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_filename = f"batch_{timestamp}_{i+1}_{os.path.basename(image_path)}"
                result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                
                # Collect result data
                result_data = {
                    'filename': os.path.basename(image_path),
                    'detections_count': len(detections),
                    'detections': [
                        {
                            'confidence': float(det['confidence']),
                            'area': int(det['area']),
                            'center': [int(x) for x in det.get('center', [0, 0])]
                        } for det in detections
                    ],
                    'result_image': f"/result/{result_filename}",
                    'status': 'success' if detections else 'no_detection'
                }
                results.append(result_data)
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                results.append({
                    'filename': os.path.basename(image_path),
                    'detections_count': 0,
                    'detections': [],
                    'result_image': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results

# Initialize detector
detector = RimDetector(crop_model, mask_generator)

@app.route('/')
def index():
    """Main page"""
    return render_template('rim_test.html')

@app.route('/api/detect', methods=['POST'])
def detect_rims():
    """Detect rims in uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        detection_type = request.form.get('type', 'auto')  # auto, full, crop
        replacement_mode = request.form.get('replacement_mode', 'false').lower() == 'true'
        
        # Save uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Handle replacement image if provided
        replacement_image = None
        if replacement_mode and 'replacement_image' in request.files:
            repl_file = request.files['replacement_image']
            if repl_file.filename != '':
                # Load replacement image
                repl_filename = f"repl_{timestamp}_{repl_file.filename}"
                repl_filepath = os.path.join(app.config['UPLOAD_FOLDER'], repl_filename)
                repl_file.save(repl_filepath)
                replacement_image = cv2.imread(repl_filepath)
                replacement_image = cv2.cvtColor(replacement_image, cv2.COLOR_BGR2RGB)
                print(f"✅ Replacement image loaded: {repl_filename}, shape: {replacement_image.shape}")
            else:
                print("❌ No replacement image file provided")
        elif replacement_mode:
            print("❌ Replacement mode enabled but no replacement_image in request.files")
        else:
            print("ℹ️ Replacement mode disabled")
        
        # Detect based on type
        if detection_type == 'crop':
            # Direct crop detection
            detections, image = detector.detect_rim_crop_only(filepath)
            is_full_image = False
        else:
            # Full pipeline (auto or full)
            detections, image = detector.detect_rims_full_pipeline(filepath)
            is_full_image = True
        
        # Create result visualization
        result_image = detector.visualize_results(image, detections, is_full_image, replacement_mode, replacement_image)
        
        # Save result
        mode_suffix = "_replaced" if replacement_mode else ""
        result_filename = f"result{mode_suffix}_{filename}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        # Prepare response
        response_data = {
            'success': True,
            'detection_type': detection_type,
            'replacement_mode': replacement_mode,
            'detections_count': len(detections),
            'detections': [
                {
                    'id': int(i),
                    'confidence': float(det['confidence']),
                    'box': [int(x) for x in det['box']],
                    'area': int(det['area']),
                    'center': [int(x) for x in det.get('center', [0, 0])],
                    'wheel_region_id': int(det.get('wheel_region_id', -1))
                } for i, det in enumerate(detections)
            ],
            'result_image': f"/result/{result_filename}",
            'timestamp': timestamp
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_detect', methods=['POST'])
def batch_detect_rims():
    """Batch detect rims in multiple uploaded images"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images uploaded'}), 400
        
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        detection_type = request.form.get('type', 'auto')
        
        # Save uploaded files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_paths = []
        
        for i, file in enumerate(files):
            if file.filename == '':
                continue
                
            filename = f"batch_{timestamp}_{i+1}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            saved_paths.append(filepath)
        
        if not saved_paths:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        # Process batch
        batch_results = detector.process_batch_images(saved_paths, detection_type)
        
        # Calculate summary statistics
        total_images = len(batch_results)
        successful_detections = len([r for r in batch_results if r['status'] == 'success'])
        total_rims = sum(r['detections_count'] for r in batch_results)
        
        response_data = {
            'success': True,
            'detection_type': detection_type,
            'batch_summary': {
                'total_images': total_images,
                'successful_detections': successful_detections,
                'total_rims_detected': total_rims,
                'success_rate': f"{(successful_detections/total_images*100):.1f}%"
            },
            'results': batch_results,
            'timestamp': timestamp
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/result/<filename>')
def serve_result(filename):
    """Serve result image"""
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))

@app.route('/upload/<filename>')
def serve_upload(filename):
    """Serve uploaded image"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    print("🎯 Rim Detection Web Test App")
    print("=" * 50)
    print(f"📊 Crop Model: {'✅ Ready' if crop_model else '❌ Failed'}")
    print(f"🔍 SAM2 (Step1): {'✅ Ready' if mask_generator else '❌ Failed'}")
    print("🌐 http://localhost:5006")
    print("📋 Features:")
    print("   • Full car images: Step1 (wheel detection) + Step2 (rim segmentation)")
    print("   • Crop images: Direct rim segmentation")
    print("   • Auto detection type selection")
    app.run(debug=True, host='0.0.0.0', port=5006) 