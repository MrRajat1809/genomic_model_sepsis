# Use official PyTorch image with CUDA support (CUDA 12.1 is highly stable and backwards compatible)
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies, including R
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    libxml2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    r-base \
    && rm -rf /var/lib/apt/lists/*

# Install Bioconductor and GEOquery in R
RUN Rscript -e "install.packages('BiocManager', repos='http://cran.us.r-project.org')" \
    && Rscript -e "BiocManager::install(c('GEOquery', 'limma', 'Biobase'))"

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose Jupyter port
EXPOSE 8888

# Start JupyterLab by default
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]