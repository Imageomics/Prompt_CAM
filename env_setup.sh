
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir

## timm
pip install timm==1.0.24 --no-cache-dir
pip uninstall -y numpy typing-extensions
pip install numpy==1.24.3 typing-extensions==4.5.0 --no-cache-dir

pip install tensorflow==2.13.1 --no-cache-dir
pip install tensorflow-datasets==4.9.2 --no-cache-dir
pip install tensorflow-addons==0.21.0 --no-cache-dir
pip install opencv-python --no-cache-dir

## CLIP
pip install git+https://github.com/openai/CLIP.git --no-cache-dir

####utils
pip install einops --no-cache-dir
pip install scipy --no-cache-dir
pip install dotwiz --no-cache-dir
pip install pyyaml --no-cache-dir
pip install tabulate  --no-cache-dir
pip install termcolor --no-cache-dir
pip install iopath --no-cache-dir
pip install scikit-learn --no-cache-dir

pip install ftfy regex tqdm --no-cache-dir
pip install pandas --no-cache-dir
pip install matplotlib --no-cache-dir
pip install ipykernel --no-cache-dir
pip install gradio --no-cache-dir
