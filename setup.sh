#!/bin/bash

echo "🎯 SAM2 Rim Segmentation Setup"
echo "================================"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p checkpoints
mkdir -p rim_dataset/{images/train,crops/train,masks/train,labels/train,approved/{images,masks,labels},final/{images,masks,annotations}}
mkdir -p crop_training/{images/{train,val},labels/{train,val}}
mkdir -p web_uploads web_results tmp

# Download SAM2 checkpoints
echo "📥 Downloading SAM2 checkpoints..."
cd checkpoints
if [ ! -f "sam2.1_hiera_large.pt" ]; then
    echo "Downloading SAM2.1 Hiera Large..."
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi
if [ ! -f "sam2.1_hiera_base_plus.pt" ]; then
    echo "Downloading SAM2.1 Hiera Base+..."
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
fi
cd ..

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Install SAM2
echo "🔧 Installing SAM2..."
cd sam2
pip install -e .
cd ..

echo "✅ Setup complete!"
echo ""
echo "🚀 Quick Start:"
echo "1. Add vehicle images to rim_dataset/images/train/"
echo "2. Run: python gr.py"
echo "3. Run: python flask_ui.py"
echo "4. Open: http://localhost:5005"
echo ""
echo "📖 For detailed instructions, see README.md" 