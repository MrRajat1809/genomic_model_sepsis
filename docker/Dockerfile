# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies (Added pandoc & image libs for tidyverse)
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    libxml2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    pandoc \
    r-base \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# IMPORTANT: Install Python dependencies FIRST so Jupyter is available for IRkernel
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Bioconductor and GEOquery
RUN Rscript -e "options(download.file.method = 'libcurl'); install.packages('BiocManager', repos='https://cloud.r-project.org/'); BiocManager::install(c('GEOquery', 'limma', 'Biobase'), ask = FALSE)"

# Install Clinical and Visualization R Packages
RUN Rscript -e "options(download.file.method = 'libcurl'); install.packages(c('tidyverse', 'survminer', 'survival', 'ggpubr', 'svglite', 'IRkernel'), repos='http://cran.rstudio.com/')"

# Link R kernel to Jupyter (Now it will work because Jupyter is already installed)
RUN Rscript -e "IRkernel::installspec(user = FALSE)"

# Expose Jupyter port
EXPOSE 8888

# Start JupyterLab by default
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]