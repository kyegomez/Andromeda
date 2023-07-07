# base image
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# avoid warnings by switching to noninteractive
ARG DEBIAN_FRONTEND=noninteractive

# make a directory for our application
RUN mkdir -p /app
WORKDIR /app

# install system-wide dependencies
RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends curl git python3-pip python3-dev

# Install PyTorch
RUN pip3 install --upgrade torch torchvision torchaudio

# Install APEX
RUN git clone https://github.com/NVIDIA/apex.git /app/apex
WORKDIR /app/apex
RUN git checkout 265b451de8ba9bfcb67edc7360f3d8772d0a8bea
RUN pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./

# Install your other dependencies...

# Copy requirements.txt and install python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Execute the accelerate config command
RUN accelerate config

# Command to run when starting the container
CMD ["accelerate", "launch", "train_distributed.py"]
