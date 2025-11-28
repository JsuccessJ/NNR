# the versions of torch and torchtext must be matched (https://pypi.org/project/torchtext)
# the CUDA version must be matched with torch-scatter (https://github.com/rusty1s/pytorch_scatter)
TORCH_VERSION=1.12.1
TORCHTEXT_VERSION=0.13.1
CUDA_VERSION=cu113

# Disable user site-packages to force installation in conda environment
export PYTHONNOUSERSITE=1

python -m pip install torch==${TORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION}
python -m pip install torchtext==${TORCHTEXT_VERSION} --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION}
python -m pip install nltk
python -m pip install numpy
python -m pip install scikit-learn
python -m pip install torch-scatter --no-build-isolation -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
