conda create -n pixeldit python=3.10 -y && conda activate pixeldit
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install 'numpy<2'
pip install 'git+https://github.com/facebookresearch/detectron2.git'


python -c "import torch; print(torch.__version__, torch.version.cuda)"   # should print 2.4.1 12.1
pip install setuptools wheel ninja      # isolation is off, so the build tools must be present in the env
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

pip install 'setuptools<70'
pip install --no-build-isolation -r requirements.txt