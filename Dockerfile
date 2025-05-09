# 必要な場合はproxy設定を追加する
# nvidia-driver >= 515 && pytorch-2.0 binary-build
ARG CUDA="11.8.0"  
ARG CUDNN="8"
ARG UBUNTU="22.04"
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU}
# rootで実行
USER root
RUN apt-get update -y && apt-get upgrade -y && apt-get autoremove -y
RUN apt-get install -y --no-install-recommends emacs python3.9 python3-pip libsndfile1-dev
RUN apt-get install -y cython3 ffmpeg libopus-dev
# unfortunatelly, we do not use CUDA 12.0
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install torchmetrics lightning einops numpy tqdm pandas scipy numba h5py seaborn wave librosa
RUN pip3 install tensorboard
RUN pip3 install torchmetrics[audio] ffmpeg pesq pyroomacoustics
RUN pip3 install transformers
RUN pip3 install peft

