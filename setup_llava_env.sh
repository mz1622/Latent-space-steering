#!/bin/bash
# Setup script for LLaVA environment

echo "Creating conda environment 'llava'..."
conda create -n llava python=3.10 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llava

echo "Installing PyTorch 2.1.2 with CUDA 11.8..."
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

echo "Installing LLaVA..."
pip install llava==1.2.2.post1

echo "Installing transformers and other dependencies..."
pip install transformers==4.36.0
pip install pillow
pip install requests
pip install tqdm
pip install pyyaml

echo "Installing evaluation dependencies..."
pip install pycocotools
pip install scikit-learn
pip install numpy
pip install matplotlib

echo "Installing utilities..."
pip install accelerate

echo "Setup complete!"
echo ""
echo "To activate this environment, run:"
echo "  conda activate llava"
echo ""
echo "To test the environment, run:"
echo "  python scripts/evaluate_multi_model.py --model-type llava --benchmark pope --debug"
