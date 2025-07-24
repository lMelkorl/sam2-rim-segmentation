#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import shutil
from pathlib import Path
import errno

# --- Paths ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMG_DIR    = os.path.join(BASE_DIR, "rim_dataset", "images", "train")
LABEL_DIR  = os.path.join(BASE_DIR, "rim_dataset", "labels", "train")
CROP_DIR   = os.path.join(BASE_DIR, "rim_dataset", "crops", "train")
MASK_DIR   = os.path.join(BASE_DIR, "rim_dataset", "masks", "train")
SAM2_PKG   = os.path.join(BASE_DIR, "sam2")
sys.path.insert(0, SAM2_PKG)

# --- Device selection ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"🖥️ Device: {device}")

# --- Build SAM2 ---
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "sam2.1_hiera_large.pt")
CONFIG_CANDIDATES = [
    os.path.join(SAM2_PKG, "configs", "sam2.1", "sam2.1_hiera_l.yaml"),
    "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2.1_hiera_l.yaml",
]

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYTHONPATH"]      = SAM2_PKG

sam2 = None
for cfg in CONFIG_CANDIDATES:
    try:
        print(f"Trying SAM2 config: {cfg}")
        sam2 = build_sam2(cfg, CHECKPOINT, device=device, apply_postprocessing=False)
        print(f"✅ Loaded SAM2 config: {cfg}")
        break
    except Exception as e:
        print(f"❌ Failed {cfg}: {e.__class__.__name__}: {e}")
if sam2 is None:
    raise RuntimeError("Failed to load any SAM2 config")

mask_generator = SAM2AutomaticMaskGenerator(sam2)

def get_next_image_number():
    """Klasördeki yalnızca tamamen sayıdan oluşan .jpg dosyalarının
       en yükseğini bulup +1 döndürür."""
    max_num = 0
    if os.path.isdir(IMG_DIR):
        for fn in os.listdir(IMG_DIR):
            name, ext = os.path.splitext(fn)
            if ext.lower() == '.jpg' and name.isdigit():
                num = int(name)
                if num > max_num:
                    max_num = num
    
    # Sequential numbering için: 1-119 ardışık varsa, boşluk varsa da 120'den devam et
    # Ama büyük random sayılar (13276 gibi) varsa onları sıralı sisteme sok
    sequential_max = get_processed_count()  # 1,2,3...119 gibi ardışık olanların max'ı
    
    # Final number: hem sequential max hem de max found number'ın büyüğü
    return max(max_num, sequential_max) + 1

def rename_images(new_files):
    """new_files listesini numerik artan sırayla işleyip
       ardışık numaralarla yeniden adlandırır."""
    start = get_next_image_number()
    processed = []

    # Numerik sıralama; sayısal olmayanları en sona atar.
    def numeric_key(fn):
        name = os.path.splitext(fn)[0]
        return int(name) if name.isdigit() else float('inf')

    for i, filename in enumerate(sorted(new_files, key=numeric_key), start=start):
        old_path = os.path.join(IMG_DIR, filename)
        new_filename = f"{i}.jpg"
        new_path = os.path.join(IMG_DIR, new_filename)
        try:
            with Image.open(old_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(new_path, 'JPEG', quality=95, optimize=True)

            # Eski dosyayı sil
            if old_path != new_path:
                os.remove(old_path)

            print(f"  ✅ {filename} → {new_filename}")
            processed.append(new_filename)
        except Exception as e:
            print(f"  ❌ {filename} işlenirken hata: {e}")

    return processed




def ensure_writable_dir(path):
    """
    path yazılabilir değilse (Errno 30), working directory içinde
    aynı ada bir klasör oluşturup onunla devam eder.
    """
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except OSError as e:
        if e.errno == errno.EROFS:
            # Salt‑okunur dosya sistemi → fallback
            fallback = os.path.join(os.getcwd(), os.path.basename(path.rstrip('/')))
            os.makedirs(fallback, exist_ok=True)
            print(f"🔔 '{path}' yazılamıyor. Fallback olarak '{fallback}' kullanılıyor.")
            return fallback
        else:
            raise

def get_processed_count():
    """
    Klasörde 1.jpg, 2.jpg, 3.jpg, … ardışık giden
    blok varsa en büyük n değerini döndürür.
    Örn: 1–119 var, 120 yok, 121–200 de olsa → 119 döner.
    """
    nums = {
        int(fn.split('.')[0])
        for fn in os.listdir(IMG_DIR)
        if fn.lower().endswith('.jpg') and fn.split('.')[0].isdigit()
    }
    n = 0
    while (n + 1) in nums:
        n += 1
    return n

def preprocess_new_images():
    """
    Yeni eklenen görselleri ardışık numaralarla yeniden adlandırır.
    Salt‑okunur IMG_DIR için fallback dizin kullanır.
    """
    global IMG_DIR
    IMG_DIR = ensure_writable_dir(IMG_DIR)

    supported = ('.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp', '.tiff')
    processed_count = get_processed_count()

    # “Yeni” dosyaları topla
    new_files = []
    for fn in os.listdir(IMG_DIR):
        low = fn.lower()
        if not low.endswith(supported):
            continue
        name, _ = os.path.splitext(fn)
        
        # Sequential range içindeki dosyaları atla (1,2,3...119 gibi)
        if name.isdigit() and int(name) <= processed_count:
            continue
            
        # Sequential range dışındaki tüm dosyaları "yeni" kabul et
        # Bu büyük random sayıları (13276.jpg) ve text-based isimleri içerir
        new_files.append(fn)

    if not new_files:
        print("📂 Yeni görsel bulunamadı – mevcut numaralı görseller kullanılacak")
        return []

    print(f"📥 {len(new_files)} yeni görsel bulundu, yeniden adlandırılıyor...")

    # Sayısalsalara öncelik ver: önce kendi içlerinde küçükten büyüğe, sonra diğerleri
    def key_fn(fn):
        name, _ = os.path.splitext(fn)
        if name.isdigit():
            return (0, int(name))
        else:
            return (1, fn.lower())

    new_files = sorted(new_files, key=key_fn)

    # Ardışık atama
    next_num = processed_count + 1
    processed_list = []

    for fn in new_files:
        old_path = os.path.join(IMG_DIR, fn)
        new_name = f"{next_num}.jpg"
        new_path = os.path.join(IMG_DIR, new_name)

        try:
            with Image.open(old_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(new_path, 'JPEG', quality=95, optimize=True)

            # Eski dosyayı kaldır
            os.remove(old_path)

            print(f"  ✅ {fn} → {new_name}")
            processed_list.append(new_name)
            next_num += 1

        except Exception as e:
            print(f"  ❌ {fn} işlenirken hata: {e}")

    return processed_list

def process_images():
    """
    Tüm iş akışınızı buraya eklediğinizde, en başta yeni dosyaları
    preprocess_new_images() ile yeniden adlandırıp,
    sonra aşağıdaki istatistikleri yazdırabilirsiniz.
    """
    # … (adım 1: SAM2 model yükleme, loglar vs.)

    # ➊ İşlem öncesi kaç yeni dosya var?
    supported = ('.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp', '.tiff')
    initial_new = [
        fn for fn in os.listdir(IMG_DIR)
        if fn.lower().endswith(supported) and not os.path.splitext(fn)[0].isdigit()
    ]
    initial_new_count = len(initial_new)

    # ➋ Yeniden adlandır ve işlenen listeyi al
    processed_files = preprocess_new_images()
    processed_new_count = len(processed_files)

    # ➌ Şu anki toplam işlenmiş sayısı
    total_processed = get_processed_count()

    # ➍ Bekleyen dosya sayısı (negatif olmaz)
    pending = max(0, initial_new_count - processed_new_count)

    # ➎ İstatistikleri yazdır
    print("\n============================================================")
    print("📊 DATASET İSTATİSTİKLERİ")
    print("============================================================")
    print(f"📷 Toplam Görsel (işlenmiş): {total_processed}")
    print(f"✅ Yeni İşlenen Görsel:       {processed_new_count}")
    print(f"⏳ Bekleyen Görsel:           {pending}")
    print("============================================================\n")



def get_unprocessed_images():
    """Henüz işlenmemiş görselleri bul"""
    if not os.path.exists(IMG_DIR):
        return []
    
    # Tüm numaralı jpg dosyalarını bul
    all_images = []
    for filename in os.listdir(IMG_DIR):
        if filename.lower().endswith('.jpg'):
            try:
                num = int(os.path.splitext(filename)[0])
                all_images.append(filename)
            except ValueError:
                continue
    
    # İşlenmiş olanları kontrol et (mask dosyaları var mı?)
    processed = set()
    if os.path.exists(MASK_DIR):
        for filename in os.listdir(MASK_DIR):
            if '_mask.png' in filename:
                # 1_1_coarse_mask.png -> 1_1 -> 1.jpg
                base_name = filename.split('_')[0]
                processed.add(f"{base_name}.jpg")
    
    # İşlenmemiş olanları döndür
    unprocessed = [img for img in all_images if img not in processed]
    return sorted(unprocessed, key=lambda x: int(os.path.splitext(x)[0]))

# --- Helpers ---
def load_image(path, max_dim=1024):
    img = Image.open(path)
    w,h = img.size
    scale = min(max_dim/max(w,h), 1.0)
    img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    return np.array(img.convert("RGB"))

def filter_rims(masks, min_area=1500, circ_thresh=0.65):
    rims = []
    for ann in masks:
        seg  = ann["segmentation"].astype(np.uint8)
        area = ann["area"]
        if area < min_area: continue

        cnts,_ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)

        peri = cv2.arcLength(cnt, True)
        circ = 4*np.pi*area/(peri*peri+1e-6)
        if circ < circ_thresh: continue

        x,y,w,h = cv2.boundingRect(cnt)
        ar = w/h
        if not (0.8 <= ar <= 1.2): continue

        rims.append((circ, area, (x,y,x+w,y+h)))

    if len(rims) < 2:
        return [box for _,_,box in rims]

    candidates = []
    for i in range(len(rims)):
        for j in range(i+1, len(rims)):
            _,a1,b1 = rims[i]
            _,a2,b2 = rims[j]
            x1,y1,x2,y2 = b1
            x3,y3,x4,y4 = b2
            dx = abs((x1+x2)/2 - (x3+x4)/2)
            dy = abs((y1+y2)/2 - (y3+y4)/2)
            size_sim = min(a1,a2)/max(a1,a2)
            if size_sim>0.6 and dx>dy:
                score = dx*size_sim
                candidates.append((score,b1,b2))
    if candidates:
        best = max(candidates, key=lambda x:x[0])
        return [best[1], best[2]]

    return [box for _,_,box in sorted(rims, key=lambda x:-x[0])[:2]]

def refine_mask_with_hough_and_roi(crop):
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 2)
    edges = cv2.Canny(blur, 50, 150)
    min_r = int(0.3 * min(w,h))
    max_r = int(0.6 * min(w,h))
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h/2,
                               param1=50, param2=30, minRadius=min_r, maxRadius=max_r)
    if circles is not None:
        circs = np.round(circles[0]).astype(int)
        cx0, cy0 = w//2, h//2
        cx, cy, r = min(circs, key=lambda c: np.hypot(c[0]-cx0, c[1]-cy0))
        Y, X = np.ogrid[:h, :w]
        circ_mask = (X-cx)**2 + (Y-cy)**2 <= r*r
        return (circ_mask.astype(np.uint8)*255)

    masks = mask_generator.generate(crop)
    best_mask, best_score = None, 0
    for m in masks:
        seg  = m["segmentation"].astype(np.uint8)
        area = m["area"]
        cnts,_ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        circ = 4*np.pi*area/(peri*peri+1e-6)
        score = circ * area
        if score > best_score:
            best_score = score
            best_mask  = seg
    return best_mask

def save_yolo_labels(boxes, W, H, out_txt):
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt,"w") as f:
        for x0,y0,x1,y1 in boxes:
            cx = (x0+x1)/2 / W
            cy = (y0+y1)/2 / H
            bw = (x1-x0)/W
            bh = (y1-y0)/H
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def generate_three_mask_variants(crop):
    """
    3 farklı maskeleme yaklaşımı:
    1. Coarse SAM2 maskesi (direkt generate ile gelen)
    2. Hough Circle maskesi (daireye dayalı) 
    3. ROI tabanlı SAM2 refinement
    """
    h, w = crop.shape[:2]
    variants = {}
    
    # --- Approach 1: Coarse SAM2 Mask ---
    crop_masks = mask_generator.generate(crop)
    coarse_mask = np.zeros((h, w), dtype=np.uint8)
    if crop_masks:
        # Select the largest area
        best_ann = max(crop_masks, key=lambda x: x['area'])
        coarse_mask = best_ann['segmentation'].astype(np.uint8) * 255
    
    variants['coarse'] = coarse_mask
    
    # --- Approach 2: Hough Circle Mask ---
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blur, 50, 150)
    min_r = int(0.3 * min(w, h))
    max_r = int(0.6 * min(w, h))
    
    hough_mask = np.zeros((h, w), dtype=np.uint8)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h/2,
                               param1=50, param2=30, minRadius=min_r, maxRadius=max_r)
    
    if circles is not None:
        circs = np.round(circles[0]).astype(int)
        cx0, cy0 = w//2, h//2
        # Select the circle closest to center
        cx, cy, r = min(circs, key=lambda c: np.hypot(c[0]-cx0, c[1]-cy0))
        cv2.circle(hough_mask, (cx, cy), r, 255, thickness=-1)
    
    variants['hough'] = hough_mask
    
    # --- Approach 3: ROI-based SAM2 Refinement ---
    refined_mask = np.zeros((h, w), dtype=np.uint8)
    if crop_masks:
        best_mask, best_score = None, 0
        for ann in crop_masks:
            seg = ann['segmentation'].astype(np.uint8)
            area = ann['area']
            
            # Circularity and area score calculation
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
                
            cnt = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
                
            circ = 4*np.pi*area/(peri*peri + 1e-6)
            
            # Successful parameters from example_code.py
            if circ >= 0.6 and area >= 1200:  # Minimum circularity and area
                score = circ * area
                if score > best_score:
                    best_score = score
                    best_mask = seg
        
        if best_mask is not None:
            refined_mask = best_mask * 255
    
    variants['refined'] = refined_mask
    
    return variants

def create_overlay_variants(crop, mask_variants):
    """Her maske varyantı için overlay görsel oluştur"""
    overlays = {}
    colors = {
        'coarse': [255, 0, 0],    # Red
        'hough': [0, 255, 0],     # Green  
        'refined': [0, 0, 255]    # Blue
    }
    
    for variant_name, mask in mask_variants.items():
        overlay = crop.copy()
        if mask.sum() > 0:  # If mask is not empty
            colored = np.zeros_like(overlay)
            colored[:,:] = colors[variant_name]
            mask_bool = (mask > 127).astype(bool)
            
            # Alpha blending
            overlay[mask_bool] = cv2.addWeighted(
                overlay[mask_bool], 0.6, 
                colored[mask_bool], 0.4, 0
            )
        
        overlays[variant_name] = overlay
    
    return overlays

def save_crops_and_mask_variants(image, boxes, crop_dir, mask_dir, base):
    """Kırpılmış görselleri ve 3 maske varyantını kaydet"""
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    for i, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        crop = image[y0:y1, x0:x1]
        crop_name = f"{base}_{i}"
        
        # Save cropped image
        crop_path = os.path.join(crop_dir, f"{crop_name}.jpg")
        Image.fromarray(crop).save(crop_path)
        
        # 3 farklı maske varyantı oluştur
        mask_variants = generate_three_mask_variants(crop)
        
        # Save mask and overlay for each variant
        for variant_name, mask in mask_variants.items():
            # Save binary mask
            mask_path = os.path.join(mask_dir, f"{crop_name}_{variant_name}_mask.png")
            cv2.imwrite(mask_path, mask)
            
            # Create and save overlay image
            if mask.sum() > 0:
                overlay = crop.copy()
                colored = np.zeros_like(overlay)
                
                # Select color based on variant
                if variant_name == 'coarse':
                    colored[:,:,0] = 255  # Red
                elif variant_name == 'hough': 
                    colored[:,:,1] = 255  # Green
                else:  # refined
                    colored[:,:,2] = 255  # Blue
                
                mask_bool = (mask > 127).astype(bool)
                overlay[mask_bool] = cv2.addWeighted(
                    overlay[mask_bool], 0.6,
                    colored[mask_bool], 0.4, 0
                )
            else:
                overlay = crop.copy()
            
            overlay_path = os.path.join(mask_dir, f"{crop_name}_{variant_name}_overlay.png")
            Image.fromarray(overlay).save(overlay_path)
        
        print(f"  └─ {crop_name}: 3 maske varyantı (coarse, hough, refined) kaydedildi")


# --- Main processing loop ---
def process_images():
    """Main processing loop - process new images"""
    
    # Create directories
    for dir_path in [LABEL_DIR, CROP_DIR, MASK_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # First organize new images with proper names
    new_files = preprocess_new_images()
    if new_files:
        print(f"✅ {len(new_files)} new images organized")
    
    # Find unprocessed images
    unprocessed_images = get_unprocessed_images()
    
    if not unprocessed_images:
        print("🎉 All images already processed!")
        return
    
    print(f"🔄 {len(unprocessed_images)} images to process...")
    
    for fn in unprocessed_images:
        base = os.path.splitext(fn)[0]
        img_p = os.path.join(IMG_DIR, fn)
        
        print(f"🔄 Processing {fn}...")
        
        try:
            image = load_image(img_p)
            H, W = image.shape[:2]

            masks = mask_generator.generate(image)
            boxes = filter_rims(masks)

            if boxes:
                # Save YOLO labels
                txt_p = os.path.join(LABEL_DIR, f"{base}.txt")
                save_yolo_labels(boxes, W, H, txt_p)
                
                # Save cropped images and masks
                save_crops_and_mask_variants(image, boxes, CROP_DIR, MASK_DIR, base)
                
                print(f"✅ {fn}: {len(boxes)} rims detected → masks and labels saved")
            else:
                print(f"⚠️ {fn}: No rims detected")
                
        except Exception as e:
            print(f"❌ Error processing {fn}: {str(e)}")

def print_statistics():
    """İstatistikleri yazdır"""
    
    # Toplam görsel sayısı
    total_images = len([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]) if os.path.exists(IMG_DIR) else 0
    
    # İşlenmiş görsel sayısı
    processed_images = 0
    if os.path.exists(MASK_DIR):
        processed_bases = set()
        for f in os.listdir(MASK_DIR):
            if '_mask.png' in f:
                base = f.split('_')[0]
                processed_bases.add(base)
        processed_images = len(processed_bases)
    
    # Toplam jant sayısı
    total_rims = len([f for f in os.listdir(CROP_DIR) if f.endswith('.jpg')]) if os.path.exists(CROP_DIR) else 0
    
    print("\n" + "="*60)
    print("📊 DATASET İSTATİSTİKLERİ")
    print("="*60)
    print(f"📷 Toplam Görsel: {total_images}")
    print(f"✅ İşlenmiş Görsel: {processed_images}")
    print(f"⏳ Bekleyen Görsel: {total_images - processed_images}")
    print(f"🎯 Tespit Edilen Jant: {total_rims}")
    print(f"📁 Masks Klasörü: {MASK_DIR}")
    print(f"📁 Crops Klasörü: {CROP_DIR}")
    print("="*60)

if __name__ == "__main__":
    print("🚀 Rim Detection and Masking System v2.0")
    print("📋 Features:")
    print("  • Automatic file naming")
    print("  • Incremental processing (new images only)")
    print("  • 3 mask variants (coarse, hough, refined)")
    print("  • YOLO format labeling")
    print("-" * 60)
    
    process_images()
    print_statistics()
    
    print("\n🎉 Processing completed!")
    print("💡 Next step: Review your masks with Flask UI")
    print("   python flask_ui.py")
    print("-" * 60)
