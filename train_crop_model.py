#!/usr/bin/env python3
"""
Rim Crop Segmentation Model Training
Approved crop images ile direkt segmentation eğitimi
"""

import os
import shutil
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
import random
from pathlib import Path

class CropModelTrainer:
    def __init__(self):
        self.base_dir = Path(".")
        self.rim_dataset_dir = self.base_dir / "rim_dataset"
        self.training_dir = self.base_dir / "crop_training"
        
        # Create training directories
        self.setup_directories()
    
    def setup_directories(self):
        """Training klasörlerini oluştur"""
        dirs = [
            self.training_dir / "images" / "train",
            self.training_dir / "images" / "val", 
            self.training_dir / "labels" / "train",
            self.training_dir / "labels" / "val"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Training directory: {self.training_dir}")
    
    def analyze_approved_crops(self):
        """Approved crop'ları analiz et"""
        print("📊 Approved crops analizi...")
        
        approved_dir = self.rim_dataset_dir / "approved"
        
        if not approved_dir.exists():
            print("❌ Approved directory bulunamadı!")
            return None
        
        # Approved files
        approved_images = list((approved_dir / "images").glob("*.jpg"))
        approved_masks = list((approved_dir / "masks").glob("*.png"))
        
        print(f"   Approved images: {len(approved_images)}")
        print(f"   Approved masks: {len(approved_masks)}")
        
        # Match images with masks
        matched_samples = []
        
        for img_path in approved_images:
            base_name = img_path.stem
            mask_path = approved_dir / "masks" / f"{base_name}_mask.png"
            
            if mask_path.exists():
                matched_samples.append({
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'name': base_name
                })
        
        print(f"   Matched samples: {len(matched_samples)}")
        return matched_samples
    
    def create_crop_dataset(self, matched_samples):
        """Crop dataset oluştur"""
        print("🔄 Crop YOLO dataset oluşturuluyor...")
        
        # Shuffle samples
        random.shuffle(matched_samples)
        
        train_count = 0
        val_count = 0
        processed_count = 0
        
        for i, sample in enumerate(matched_samples):
            success = self.process_crop_sample(sample, i)
            
            if success:
                processed_count += 1
                
                # 80/20 train/val split
                is_val = (i % 5 == 0)
                split = "val" if is_val else "train"
                
                if is_val:
                    val_count += 1
                else:
                    train_count += 1
                
                # Copy image
                target_img = self.training_dir / "images" / split / f"{processed_count:04d}.jpg"
                shutil.copy2(sample['image_path'], target_img)
                
                if processed_count % 50 == 0:
                    print(f"   ✅ Processed: {processed_count}/{len(matched_samples)}")
        
        print(f"🎉 Crop dataset oluşturuldu!")
        print(f"   Training: {train_count} images")
        print(f"   Validation: {val_count} images")
        print(f"   Total: {processed_count} images")
        
        return train_count, val_count, processed_count
    
    def process_crop_sample(self, sample, index):
        """Tek crop sample'ı işle"""
        try:
            # Load image and mask
            image = cv2.imread(str(sample['image_path']))
            mask = cv2.imread(str(sample['mask_path']), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                return False
            
            h, w = image.shape[:2]
            
            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return False
            
            # Get main contour
            main_contour = max(contours, key=cv2.contourArea)
            
            # Normalize contour points
            normalized_points = []
            
            for point in main_contour:
                px, py = point[0]
                
                # Normalize to [0, 1]
                norm_x = max(0, min(1, px / w))
                norm_y = max(0, min(1, py / h))
                
                normalized_points.extend([norm_x, norm_y])
            
            if len(normalized_points) < 6:  # Need at least 3 points
                return False
            
            # Create YOLO annotation
            yolo_annotation = f"0 {' '.join(map(str, normalized_points))}"
            
            # Write annotation file
            split = "val" if index % 5 == 0 else "train"
            label_file = self.training_dir / "labels" / split / f"{index + 1:04d}.txt"
            
            with open(label_file, 'w') as f:
                f.write(yolo_annotation)
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error processing {sample['name']}: {e}")
            return False
    
    def create_dataset_yaml(self):
        """Dataset YAML configuration oluştur"""
        dataset_config = {
            'path': str(self.training_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['rim']
        }
        
        yaml_path = self.training_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"📄 Dataset YAML: {yaml_path}")
        return yaml_path
    
    def train_model(self, dataset_yaml, model_size='m'):
        """YOLO model eğitimi"""
        print(f"🚀 YOLOv8{model_size}-seg model eğitimi başlıyor...")
        
        # Model seç - Medium model for better accuracy
        model_file = f'yolov8{model_size}-seg.pt'
        model = YOLO(model_file)
        
        # Training parameters
        results = model.train(
            data=str(dataset_yaml),
            epochs=200,             # More epochs for better convergence
            imgsz=512,              # Good size for rim crops
            batch=32,               # Larger batch for crops
            patience=30,            # More patience
            save=True,
            verbose=True,
            name=f'rim_crop_segmentation_v8{model_size}',
            project='runs/segment',
            
            # Optimization
            lr0=0.01,               # Good learning rate
            weight_decay=0.0005,    # Regularization
            augment=True,           # Data augmentation
            cos_lr=True,            # Cosine scheduler
            
            # Advanced settings
            amp=True,               # Mixed precision
            mosaic=0.8,             # Mosaic augmentation
            mixup=0.15,             # Mixup augmentation
            copy_paste=0.1,         # Copy-paste augmentation
            
            # Quality settings
            val=True,               # Validation
            plots=True,             # Training plots
            save_period=10,         # Save every 10 epochs
        )
        
        model_path = f"runs/segment/rim_crop_segmentation_v8{model_size}/weights/best.pt"
        print(f"🎉 Model eğitimi tamamlandı!")
        print(f"📁 Best model: {model_path}")
        
        return results, model_path

def main():
    """Ana eğitim pipeline"""
    print("🎯 Rim Crop Segmentation Model Training")
    print("=" * 50)
    
    trainer = CropModelTrainer()
    
    # 1. Approved crops analizi
    matched_samples = trainer.analyze_approved_crops()
    
    if not matched_samples:
        print("❌ No approved samples found!")
        return
    
    # 2. Crop dataset oluştur
    train_count, val_count, total_count = trainer.create_crop_dataset(matched_samples)
    
    if total_count == 0:
        print("❌ No training data created!")
        return
    
    # 3. Dataset YAML oluştur
    dataset_yaml = trainer.create_dataset_yaml()
    
    # 4. Model eğitimi
    print(f"\n🎯 {train_count} train + {val_count} val crop samples ile eğitim...")
    
    # YOLOv8m kullan (better accuracy for crops)
    results, model_path = trainer.train_model(dataset_yaml, model_size='m')
    
    print("\n🏆 BAŞARILI! Crop model eğitimi tamamlandı!")
    print(f"📊 Eğitilen model: {model_path}")
    print("🔍 Test etmek için:")
    print(f"   yolo predict model={model_path} source='rim_crop.jpg'")
    print("\n💡 Bu model crop'larda çalışır, full car images için Step1+Step2 gerekli!")

if __name__ == "__main__":
    main() 