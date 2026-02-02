@echo off
REM Setup script for LLaVA environment

echo Creating conda environment 'llava'...
conda create -n llava python=3.10 -y

echo Activating environment...
call conda activate llava

echo Installing PyTorch 2.1.2 with CUDA 11.8...
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

echo Installing NumPy 1.x (required for compatibility)...
pip install "numpy<2"

echo Installing core dependencies...
pip install transformers==4.36.0
pip install pillow
pip install requests
pip install tqdm
pip install pyyaml
pip install accelerate
pip install protobuf
pip install sentencepiece

echo Installing evaluation dependencies...
pip install pycocotools-windows
pip install scikit-learn
pip install matplotlib

echo Installing LLaVA from GitHub...
pip install git+https://github.com/haotian-liu/LLaVA.git

echo Setup complete!
echo.
echo To activate this environment, run:
echo   conda activate llava
echo.
echo To test the environment, run:
echo   python scripts/evaluate_multi_model.py --model-type llava --benchmark pope --debug
