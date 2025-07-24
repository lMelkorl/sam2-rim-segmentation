#!/usr/bin/env python3
"""
Flask UI for Rim Mask Selection v3.0
Complete dataset management with approved mask organization
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from PIL import Image
import base64
from io import BytesIO
import shutil
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROP_DIR = os.path.join(BASE_DIR, "rim_dataset", "crops", "train")
MASK_DIR = os.path.join(BASE_DIR, "rim_dataset", "masks", "train")
LABELS_DIR = os.path.join(BASE_DIR, "rim_dataset", "labels", "train")
SELECTION_FILE = os.path.join(BASE_DIR, "rim_dataset", "user_selections.json")
APPROVED_FILE = os.path.join(BASE_DIR, "rim_dataset", "approved_samples.json")

# Approved dataset paths
APPROVED_DIR = os.path.join(BASE_DIR, "rim_dataset", "approved")
APPROVED_CROPS_DIR = os.path.join(APPROVED_DIR, "images")
APPROVED_MASKS_DIR = os.path.join(APPROVED_DIR, "masks")
APPROVED_LABELS_DIR = os.path.join(APPROVED_DIR, "labels")

# Final dataset paths for training
FINAL_DIR = os.path.join(BASE_DIR, "rim_dataset", "final")
FINAL_IMAGES_DIR = os.path.join(FINAL_DIR, "images")
FINAL_MASKS_DIR = os.path.join(FINAL_DIR, "masks")
FINAL_ANNOTATIONS_DIR = os.path.join(FINAL_DIR, "annotations")

def create_directories():
    """Create necessary directories"""
    dirs_to_create = [
        APPROVED_DIR, APPROVED_CROPS_DIR, APPROVED_MASKS_DIR, APPROVED_LABELS_DIR,
        FINAL_DIR, FINAL_IMAGES_DIR, FINAL_MASKS_DIR, FINAL_ANNOTATIONS_DIR
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

def cleanup_non_approved_masks(sample_name, approved_variant):
    """Onaylanmayan varyantları temizle"""
    variants = ['coarse', 'hough', 'refined', 'manual']
    for variant in variants:
        if variant != approved_variant:
            # Onaylanmayan maskeleri sil
            mask_file = os.path.join(MASK_DIR, f"{sample_name}_{variant}_mask.png")
            overlay_file = os.path.join(MASK_DIR, f"{sample_name}_{variant}_overlay.png")
            
            if os.path.exists(mask_file):
                os.remove(mask_file)
            if os.path.exists(overlay_file):
                os.remove(overlay_file)

def move_to_approved(sample_name, variant):
    """Onaylanmış örneği approved klasörüne taşı"""
    # Kaynak dosyalar
    crop_src = os.path.join(CROP_DIR, f"{sample_name}.jpg")
    # Manuel maske kontrolü
    if variant == 'manual':
        mask_src = os.path.join(MASK_DIR, f"{sample_name}_manual_mask.png")
    else:
        mask_src = os.path.join(MASK_DIR, f"{sample_name}_{variant}_mask.png")
    
    label_src = os.path.join(LABELS_DIR, f"{sample_name.split('_')[0]}.txt")
    
    # Hedef dosyalar
    crop_dst = os.path.join(APPROVED_CROPS_DIR, f"{sample_name}.jpg")
    mask_dst = os.path.join(APPROVED_MASKS_DIR, f"{sample_name}_mask.png")
    label_dst = os.path.join(APPROVED_LABELS_DIR, f"{sample_name}.txt")
    
    try:
        # Dosyaları taşı
        if os.path.exists(crop_src):
            shutil.copy2(crop_src, crop_dst)
        if os.path.exists(mask_src):
            shutil.copy2(mask_src, mask_dst)
        if os.path.exists(label_src):
            shutil.copy2(label_src, label_dst)
        
        # Diğer varyantları temizle (manuel dahil)
        cleanup_non_approved_masks(sample_name, variant)
        
        return True
    except Exception as e:
        print(f"Error moving {sample_name}: {e}")
        return False

def create_segmentation_annotations(sample_name, mask_path):
    """Segmentation modeli için annotation dosyası oluştur"""
    try:
        # Maskeyi yükle
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
            
        # Kontürları bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # En büyük kontürü seç
        main_contour = max(contours, key=cv2.contourArea)
        
        # Kontürü normalize et (0-1 arası)
        h, w = mask.shape
        normalized_contour = []
        for point in main_contour:
            x, y = point[0]
            normalized_contour.extend([x/w, y/h])
        
        # YOLO segmentation format
        annotation = f"0 {' '.join(map(str, normalized_contour))}\n"
        
        return annotation
        
    except Exception as e:
        print(f"Error creating annotation for {sample_name}: {e}")
        return None

def create_coco_annotations():
    """COCO format annotations oluştur"""
    coco_annotation = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "rim",
                "supercategory": "vehicle_part"
            }
        ]
    }
    
    annotation_id = 1
    
    # Approved klasöründeki dosyaları işle
    if os.path.exists(APPROVED_CROPS_DIR):
        for idx, filename in enumerate(sorted(os.listdir(APPROVED_CROPS_DIR))):
            if filename.endswith('.jpg'):
                image_id = idx + 1
                sample_name = filename[:-4]  # .jpg'yi kaldır
                
                # Görsel bilgilerini al
                img_path = os.path.join(APPROVED_CROPS_DIR, filename)
                with Image.open(img_path) as img:
                    width, height = img.size
                
                # Image entry
                coco_annotation["images"].append({
                    "id": image_id,
                    "file_name": filename,
                    "width": width,
                    "height": height
                })
                
                # Mask dosyasını kontrol et
                mask_path = os.path.join(APPROVED_MASKS_DIR, f"{sample_name}_mask.png")
                if os.path.exists(mask_path):
                    # Mask'i yükle ve kontür bilgilerini çıkar
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        main_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(main_contour)
                        x, y, w, h = cv2.boundingRect(main_contour)
                        
                        # Segmentation points
                        segmentation = main_contour.flatten().tolist()
                        
                        coco_annotation["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x, y, w, h],
                            "area": float(area),
                            "segmentation": [segmentation],
                            "iscrowd": 0
                        })
                        
                        annotation_id += 1
    
    return coco_annotation

# --- Helper Functions ---
def get_available_samples():
    """Mevcut kırpılmış görselleri ve varyantlarını bul"""
    samples = {}
    
    if not os.path.exists(CROP_DIR):
        return samples
        
    for filename in os.listdir(CROP_DIR):
        if filename.endswith('.jpg'):
            base_name = filename[:-4]
            
            # Varyantları kontrol et (manuel dahil)
            variants = {}
            variant_count = 0
            for variant in ['coarse', 'hough', 'refined', 'manual']:
                overlay_path = os.path.join(MASK_DIR, f"{base_name}_{variant}_overlay.png")
                mask_path = os.path.join(MASK_DIR, f"{base_name}_{variant}_mask.png")
                
                if os.path.exists(overlay_path) and os.path.exists(mask_path):
                    variants[variant] = {
                        'overlay': overlay_path,
                        'mask': mask_path
                    }
                    variant_count += 1
            
            if variants:
                samples[base_name] = {
                    'crop': os.path.join(CROP_DIR, filename),
                    'variants': variants,
                    'variant_count': variant_count
                }
    
    return samples

def load_user_selections():
    """Kullanıcı seçimlerini yükle"""
    if os.path.exists(SELECTION_FILE):
        with open(SELECTION_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_selections(selections):
    """Kullanıcı seçimlerini kaydet"""
    os.makedirs(os.path.dirname(SELECTION_FILE), exist_ok=True)
    with open(SELECTION_FILE, 'w') as f:
        json.dump(selections, f, indent=2)

def load_approved_samples():
    """Onaylanmış örnekleri yükle"""
    if os.path.exists(APPROVED_FILE):
        with open(APPROVED_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_approved_samples(approved):
    """Onaylanmış örnekleri kaydet"""
    os.makedirs(os.path.dirname(APPROVED_FILE), exist_ok=True)
    with open(APPROVED_FILE, 'w') as f:
        json.dump(approved, f, indent=2)

def get_sample_statistics():
    """Örnek istatistiklerini hesapla"""
    samples = get_available_samples()
    selections = load_user_selections()
    approved = load_approved_samples()
    
    # Approved klasöründeki dosya sayıları
    approved_count = len([f for f in os.listdir(APPROVED_CROPS_DIR) if f.endswith('.jpg')]) if os.path.exists(APPROVED_CROPS_DIR) else 0
    
    total_samples = len(samples)
    selected_samples = len(selections)
    
    # Varyant dağılımı
    variant_stats = {'coarse': 0, 'hough': 0, 'refined': 0}
    for sample_name, variant in selections.items():
        if variant in variant_stats:
            variant_stats[variant] += 1
    
    return {
        'total_samples': total_samples,
        'selected_samples': selected_samples,
        'approved_samples': approved_count,
        'completion_percent': round((selected_samples / total_samples * 100) if total_samples > 0 else 0, 1),
        'approval_percent': round((approved_count / selected_samples * 100) if selected_samples > 0 else 0, 1),
        'variant_stats': variant_stats
    }

def image_to_base64(image_path):
    """Görseli base64'e çevir"""
    try:
        with Image.open(image_path) as img:
            # Boyutu optimize et
            if img.size[0] > 400 or img.size[1] > 400:
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            buffer = BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error converting image {image_path}: {e}")
        return None

# --- Routes ---
@app.route('/')
def index():
    """Main page - list samples"""
    samples = get_available_samples()
    selections = load_user_selections()
    approved = load_approved_samples()
    stats = get_sample_statistics()
    
    sample_list = []
    for sample_name, sample_data in samples.items():
        sample_info = {
            'name': sample_name,
            'crop_path': sample_data['crop'],
            'variants': sample_data['variants'],
            'variant_count': sample_data['variant_count'],
            'selected_variant': selections.get(sample_name),
            'is_approved': sample_name in approved,
            'approval_date': approved.get(sample_name, {}).get('date') if sample_name in approved else None
        }
        sample_list.append(sample_info)
    
    # Numaraya göre sırala
    sample_list.sort(key=lambda x: int(x['name'].split('_')[0]))
    
    return render_template('index.html', 
                         samples=sample_list, 
                         stats=stats)

@app.route('/statistics')
def statistics():
    """Statistics page"""
    stats = get_sample_statistics()
    samples = get_available_samples()
    selections = load_user_selections()
    approved = load_approved_samples()
    
    # Detaylı istatistikler
    detailed_stats = {
        'total_rims_detected': len(samples),
        'by_image': {},
        'variant_performance': {
            'coarse': {'count': 0, 'percentage': 0},
            'hough': {'count': 0, 'percentage': 0},
            'refined': {'count': 0, 'percentage': 0}
        }
    }
    
    # Görsel bazında grup
    image_groups = {}
    for sample_name in samples:
        image_num = sample_name.split('_')[0]
        if image_num not in image_groups:
            image_groups[image_num] = []
        image_groups[image_num].append(sample_name)
    
    detailed_stats['by_image'] = {
        img_num: {
            'rim_count': len(rims),
            'selected_count': len([r for r in rims if r in selections]),
            'approved_count': len([r for r in rims if r in approved])
        }
        for img_num, rims in image_groups.items()
    }
    
    # Varyant performansı hesapla
    total_selected = len(selections)
    if total_selected > 0:
        for variant, count in stats['variant_stats'].items():
            detailed_stats['variant_performance'][variant] = {
                'count': count,
                'percentage': round((count / total_selected * 100), 1)
            }
    
    return render_template('statistics.html', 
                         stats=stats,
                         detailed_stats=detailed_stats,
                         current_page='statistics')

@app.route('/api/samples')
def api_samples():
    """JSON formatında örnekleri döndür"""
    samples = get_available_samples()
    selections = load_user_selections()
    approved = load_approved_samples()
    
    result = []
    for sample_name, sample_data in samples.items():
        sample_info = {
            'name': sample_name,
            'crop_image': image_to_base64(sample_data['crop']),
            'selected_variant': selections.get(sample_name, None),
            'is_approved': sample_name in approved,
            'variants': {}
        }
        
        for variant_name, variant_paths in sample_data['variants'].items():
            sample_info['variants'][variant_name] = {
                'overlay': image_to_base64(variant_paths['overlay']),
                'mask': image_to_base64(variant_paths['mask'])
            }
        
        result.append(sample_info)
    
    return jsonify(result)



@app.route('/api/approve', methods=['POST'])
def api_approve():
    """Örneği onayla ve approved klasörüne taşı"""
    data = request.get_json()
    sample_name = data.get('sample_name')
    
    if not sample_name:
        return jsonify({'error': 'sample_name gerekli'}), 400
    
    selections = load_user_selections()
    if sample_name not in selections:
        return jsonify({'error': 'Önce bir varyant seçmelisiniz'}), 400
    
    variant = selections[sample_name]
    
    # Klasörleri oluştur
    create_directories()
    
    # Approved klasörüne taşı
    if move_to_approved(sample_name, variant):
        # Onaylanmış örneklere ekle
        approved = load_approved_samples()
        approved[sample_name] = {
            'variant': variant,
            'date': datetime.now().isoformat(),
            'approved_by': 'user'
        }
        save_approved_samples(approved)
        
        return jsonify({
            'success': True,
            'sample_name': sample_name,
            'message': f'{sample_name} onaylandı ve approved klasörüne taşındı'
        })
    else:
        return jsonify({'error': 'Dosya taşıma sırasında hata oluştu'}), 500

@app.route('/api/unapprove', methods=['POST'])
def api_unapprove():
    """Onayı geri al"""
    data = request.get_json()
    sample_name = data.get('sample_name')
    
    if not sample_name:
        return jsonify({'error': 'sample_name gerekli'}), 400
    
    approved = load_approved_samples()
    if sample_name in approved:
        del approved[sample_name]
        save_approved_samples(approved)
        
        # Approved klasöründen dosyaları sil
        crop_file = os.path.join(APPROVED_CROPS_DIR, f"{sample_name}.jpg")
        mask_file = os.path.join(APPROVED_MASKS_DIR, f"{sample_name}_mask.png")
        label_file = os.path.join(APPROVED_LABELS_DIR, f"{sample_name}.txt")
        
        for file_path in [crop_file, mask_file, label_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return jsonify({
        'success': True,
        'sample_name': sample_name,
        'message': f'{sample_name} onayı geri alındı'
    })

# flask_ui.py — sadece ilgili route tanımı
@app.route('/api/save_manual_mask', methods=['POST'])
def api_save_manual_mask():
    """Save manually drawn mask - ENHANCED VERSION"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data gerekli'}), 400
            
        sample_name = data.get('sample_name')
        mask_data   = data.get('mask_data')
        if not sample_name or not mask_data:
            return jsonify({'error': 'sample_name ve mask_data gerekli'}), 400
        
        # Base64 formatını kontrol, decode ve PIL ile aç
        header, encoded = mask_data.split(',', 1)
        if not header.startswith('data:image/'):
            return jsonify({'error': 'Geçersiz image data formatı'}), 400
        mask_bytes = base64.b64decode(encoded)
        mask_img   = Image.open(BytesIO(mask_bytes)).convert('RGBA')
        mask_arr   = np.array(mask_img)
        
        # Orijinal crop resmi
        crop_path = os.path.join(CROP_DIR, f"{sample_name}.jpg")
        if not os.path.exists(crop_path):
            return jsonify({'error': f'Crop resmi bulunamadı: {sample_name}.jpg'}), 400
        crop_img = cv2.imread(crop_path)
        ch, cw   = crop_img.shape[:2]
        
        # Eğer boyutlar farklıysa nearest-neighbor ile resize
        h, w = mask_arr.shape[:2]
        if (h, w) != (ch, cw):
            mask_arr = cv2.resize(mask_arr, (cw, ch), interpolation=cv2.INTER_NEAREST)
        
        # RGBA’dan kırmızı alanı tespit, binary mask’e dönüştür
        r, g, b, a = mask_arr[:,:,0], mask_arr[:,:,1], mask_arr[:,:,2], mask_arr[:,:,3]
        red_mask   = (a>50) & (r>100) & (r>g+50) & (r>b+50)
        if not red_mask.any():
            return jsonify({'error': 'Kırmızı alan tespit edilemedi'}), 400
        binary_mask = np.where(red_mask, 255, 0).astype(np.uint8)
        
        # Kaydet: debug görüntüleri ve mask / overlay
        os.makedirs(os.path.join(MASK_DIR,'debug'), exist_ok=True)
        cv2.imwrite(os.path.join(MASK_DIR,'debug',f"{sample_name}_binary.png"), binary_mask)
        cv2.imwrite(os.path.join(MASK_DIR,f"{sample_name}_manual_mask.png"), binary_mask)
        
        overlay = crop_img.copy()
        ov_mask = binary_mask>127
        overlay[ov_mask] = cv2.addWeighted(
            overlay[ov_mask], 0.6,
            np.full_like(overlay[ov_mask],[0,0,255]),0.4,0
        )
        cv2.imwrite(os.path.join(MASK_DIR,f"{sample_name}_manual_overlay.png"), overlay)
        
        # Seçimi kaydet
        sels = load_user_selections()
        sels[sample_name] = 'manual'
        save_user_selections(sels)
        
        return jsonify({
            'success': True,
            'sample_name': sample_name,
            'message': f'Manuel maske kaydedildi ({int(red_mask.sum())} piksel)',
            'stats': {
                'canvas_size': f"{w}x{h}",
                'crop_size':   f"{cw}x{ch}"
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/select', methods=['POST'])
def api_select():
    """Save user selection (supports null value for reset)"""
    data = request.get_json()
    sample_name = data.get('sample_name')
    variant = data.get('variant')
    
    if not sample_name:
        return jsonify({'error': 'sample_name gerekli'}), 400
    
    selections = load_user_selections()
    
    if variant is None:
        # Seçimi sıfırla
        if sample_name in selections:
            del selections[sample_name]
        save_user_selections(selections)
        return jsonify({
            'success': True,
            'sample_name': sample_name,
            'message': f'{sample_name} seçimi sıfırlandı'
        })
    else:
        # Normal seçim
        selections[sample_name] = variant
        save_user_selections(selections)
        return jsonify({
            'success': True, 
            'sample_name': sample_name, 
            'variant': variant,
            'message': f'{sample_name} için {variant} varyantı seçildi'
        })

@app.route('/api/export')
def api_export():
    """Onaylanmış maskeleri final dataset olarak export et"""
    approved = load_approved_samples()
    
    if not approved:
        return jsonify({'error': 'Henüz onaylanmış örnek yok'}), 400
    
    # Final klasörlerini oluştur
    os.makedirs(FINAL_CROPS_DIR, exist_ok=True)
    os.makedirs(FINAL_MASKS_DIR, exist_ok=True)
    
    exported_count = 0
    errors = []
    
    for sample_name, approval_data in approved.items():
        selected_variant = approval_data['variant']
        
        # Kaynak dosyaları
        crop_src = os.path.join(CROP_DIR, f"{sample_name}.jpg")
        mask_src = os.path.join(MASK_DIR, f"{sample_name}_{selected_variant}_mask.png")
        
        # Hedef dosyaları
        crop_dst = os.path.join(FINAL_CROPS_DIR, f"{sample_name}.jpg")
        mask_dst = os.path.join(FINAL_MASKS_DIR, f"{sample_name}_mask.png")
        
        try:
            if os.path.exists(crop_src) and os.path.exists(mask_src):
                shutil.copy2(crop_src, crop_dst)
                shutil.copy2(mask_src, mask_dst)
                exported_count += 1
            else:
                errors.append(f"{sample_name}: Kaynak dosyalar bulunamadı")
        except Exception as e:
            errors.append(f"{sample_name}: {str(e)}")
    
    # Export bilgisini kaydet
    export_info = {
        'export_date': datetime.now().isoformat(),
        'exported_count': exported_count,
        'total_approved': len(approved),
        'errors': errors
    }
    
    with open(os.path.join(FINAL_DIR, 'export_info.json'), 'w') as f:
        json.dump(export_info, f, indent=2)
    
    return jsonify({
        'success': True,
        'exported_count': exported_count,
        'total_approved': len(approved),
        'final_dir': FINAL_DIR,
        'errors': errors
    })

@app.route('/api/create_training_dataset')
def api_create_training_dataset():
    """Model eğitimi için final dataset oluştur"""
    create_directories()
    
    try:
        # Approved dosyaları final klasörüne kopyala
        exported_count = 0
        
        if os.path.exists(APPROVED_CROPS_DIR):
            for filename in os.listdir(APPROVED_CROPS_DIR):
                if filename.endswith('.jpg'):
                    sample_name = filename[:-4]
                    
                    # Dosyaları kopyala
                    crop_src = os.path.join(APPROVED_CROPS_DIR, filename)
                    mask_src = os.path.join(APPROVED_MASKS_DIR, f"{sample_name}_mask.png")
                    
                    crop_dst = os.path.join(FINAL_IMAGES_DIR, filename)
                    mask_dst = os.path.join(FINAL_MASKS_DIR, f"{sample_name}_mask.png")
                    
                    if os.path.exists(crop_src) and os.path.exists(mask_src):
                        shutil.copy2(crop_src, crop_dst)
                        shutil.copy2(mask_src, mask_dst)
                        
                        # YOLO segmentation annotation oluştur
                        annotation = create_segmentation_annotations(sample_name, mask_src)
                        if annotation:
                            with open(os.path.join(FINAL_ANNOTATIONS_DIR, f"{sample_name}.txt"), 'w') as f:
                                f.write(annotation)
                        
                        exported_count += 1
        
        # COCO format annotations oluştur
        coco_data = create_coco_annotations()
        with open(os.path.join(FINAL_ANNOTATIONS_DIR, 'annotations.json'), 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        # Dataset bilgi dosyası oluştur
        dataset_info = {
            'name': 'Rim Segmentation Dataset',
            'description': 'Vehicle rim segmentation dataset',
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'total_samples': exported_count,
            'classes': ['rim'],
            'formats': ['YOLO', 'COCO'],
            'structure': {
                'images/': 'Crop images',
                'masks/': 'Binary masks',
                'annotations/': 'YOLO txt + COCO json'
            }
        }
        
        with open(os.path.join(FINAL_DIR, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # YAML config dosyası (YOLO için)
        yaml_content = f"""# Rim Segmentation Dataset
path: {FINAL_DIR}
train: images
val: images  # Aynı klasör, eğitim sırasında split yapılacak

nc: 1  # number of classes
names: ['rim']  # class names
"""
        
        with open(os.path.join(FINAL_DIR, 'dataset.yaml'), 'w') as f:
            f.write(yaml_content)
        
        return jsonify({
            'success': True,
            'exported_count': exported_count,
            'final_dir': FINAL_DIR,
            'formats': ['YOLO segmentation', 'COCO'],
            'files_created': [
                'dataset_info.json',
                'dataset.yaml',
                'annotations/annotations.json',
                f'annotations/*.txt ({exported_count} files)'
            ]
        })
        
    except Exception as e:
        return jsonify({'error': f'Training dataset oluşturma hatası: {str(e)}'}), 500

@app.route('/api/stats')
def api_stats():
    """İstatistikleri JSON olarak döndür"""
    return jsonify(get_sample_statistics())

@app.route('/image/<path:filename>')
def serve_image(filename):
    """Görselleri servis et"""
    if '_mask.png' in filename or '_overlay.png' in filename:
        return send_file(os.path.join(MASK_DIR, filename))
    elif filename.endswith('.jpg'):
        return send_file(os.path.join(CROP_DIR, filename))
    else:
        return "Forbidden", 403

if __name__ == '__main__':
    # Create directories
    create_directories()
    
    print("🌐 Flask UI v3.0 - Complete Dataset Manager")
    print("📋 New Features:")
    print("  • Approved masks in separate folder")
    print("  • Other variants automatically cleaned")
    print("  • YOLO + COCO formats for model training")
    print("  • Fully automated dataset management")
    print(f"📁 Working: {CROP_DIR}")
    print(f"📁 Approved: {APPROVED_DIR}")
    print(f"📁 Training: {FINAL_DIR}")
    print("🎯 Available at http://localhost:5005")
    
    app.run(debug=True, host='0.0.0.0', port=5005) 