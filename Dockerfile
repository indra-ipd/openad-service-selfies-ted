FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS runtime

# Set the environment variables for the container system
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles \
    PYTHONUNBUFFERED=1

# Install the required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    build-essential curl git ssh libxrender1 libxext6\
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the rest of the application code to the working directory
COPY ./requirements.txt ./requirements.txt
COPY ./requirements_extra.txt ./requirements_extra.txt

# Install the required dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt
# Installing pytorch-fast-transformers in succession to pick up the torch
RUN python -m pip install --no-cache-dir -r requirements_extra.txt 

# Copy the rest of the application code to the working directory
COPY . .

# Expose the network port
EXPOSE 8080

# Set the environment variables for the application
ENV HF_HOME="/tmp/.cache/huggingface" \
    MPLCONFIGDIR="/tmp/.config/matplotlib" \
    LOGGING_CONFIG_PATH="/tmp/app.log" \
    gt4sd_local_cache_path="/tmp/.openad_models" \
    GT4SD_S3_HOST="s3.us-east-1.amazonaws.com" \
    gt4sd_s3_bucket_algorithms="ad-stage-fm4m-algorithms"\
    gt4sd_s3_bucket_properties="ad-stage-fm4m-algorithms"\
    GT4SD_S3_SECRET_KEY="" \
    GT4SD_S3_ACCESS_KEY="" \
    GT4SD_S3_HOST_HUB="s3.us-east-1.amazonaws.com" \
    GT4SD_S3_ACCESS_KEY_HUB="" \
    GT4SD_S3_SECRET_KEY_HUB="" \
    gt4sd_s3_bucket_hub_algorithms="ad-stage-fm4m-algorithms"\
    gt4sd_s3_bucket_hub_properties="ad-stage-fm4m-algorithms"\
    SELECTED_ALGORITHM_APPS="QM9-SELFIES"

# Specify the command to run when the container starts
CMD ["python", "app.py"]
