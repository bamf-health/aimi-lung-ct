FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set work dirs
RUN mkdir /app /app/data /app/data/input_data /app/data/output_data
WORKDIR /app
ENV PATH="/root/.local/bin:${PATH}"

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install system utilities and CUDA related dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt update && apt install -y --no-install-recommends \
    dcm2niix \
    wget \
    vim \
    p7zip \
    p7zip-full \
    zip \
    unzip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install python tools needed for nnUNet inference
RUN pip install --user --upgrade pip
RUN pip install --user --no-cache-dir \
    nnunet \
    pydicom \
    SimpleITK \
    dcm2niix \
    pyyaml \
    scikit-build \
    pynrrd

# Pull nnUNet model weights into the container for Task777_CT_Nodules
ENV WEIGHTS_DIR_NODULES=/root/.nnunet/nnUNet_models/nodules/nnUNet/
RUN mkdir -p $WEIGHTS_DIR_NODULES
ENV TASK_NAME_NODULES=Task777_CT_Nodules
ENV WEIGHTS_FN=$TASK_NAME_NODULES.zip
ENV WEIGHTS_URL=https://zenodo.org/record/8290147/files/$WEIGHTS_FN
RUN wget --directory-prefix ${WEIGHTS_DIR_NODULES} ${WEIGHTS_URL} --no-check-certificate
RUN unzip ${WEIGHTS_DIR_NODULES}${WEIGHTS_FN} -d ${WEIGHTS_DIR_NODULES}
RUN rm ${WEIGHTS_DIR_NODULES}${WEIGHTS_FN}

# Pull nnUNet model weights into the container for Task775_CT_NSCLC_RG
ENV WEIGHTS_DIR_NSCLC_RG=/root/.nnunet/nnUNet_models/nsclc_rg/nnUNet/
RUN mkdir -p $WEIGHTS_DIR_NSCLC_RG
ENV TASK_NAME_NSCLC_RG=Task775_CT_NSCLC_RG
ENV WEIGHTS_FN=$TASK_NAME_NSCLC_RG.zip
ENV WEIGHTS_URL=https://zenodo.org/record/8290169/files/$WEIGHTS_FN
RUN wget --directory-prefix ${WEIGHTS_DIR_NSCLC_RG} ${WEIGHTS_URL} --no-check-certificate
RUN unzip ${WEIGHTS_DIR_NSCLC_RG}${WEIGHTS_FN} -d ${WEIGHTS_DIR_NSCLC_RG}
RUN rm ${WEIGHTS_DIR_NSCLC_RG}${WEIGHTS_FN}

# specify cuda and nnunet specific environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV WEIGHTS_FOLDER_NODULES=$WEIGHTS_DIR_NODULES
ENV WEIGHTS_FOLDER_NSCLC_RG=$WEIGHTS_DIR_NSCLC_RG

# Install binaries for itkimage2segimage package
ENV PACKAGE_DIR="/root/.local/bin/"
ENV PACKAGE_TAR="dcmqi-1.2.5-linux.tar.gz"
ENV DCMQI_PACKAGE_PATH="${PACKAGE_DIR}dcmqi-1.2.5-linux"
ENV ITKIMAGE2SEGIMAGE_URL=https://github.com/QIICR/dcmqi/releases/download/v1.2.5/${PACKAGE_TAR}
RUN wget --directory-prefix ${PACKAGE_DIR} ${ITKIMAGE2SEGIMAGE_URL} --no-check-certificate
RUN tar -zxvf ${PACKAGE_DIR}${PACKAGE_TAR}
RUN rm ${PACKAGE_DIR}${PACKAGE_TAR}

# Copy scripts and config
COPY app/src/* /app/
COPY app/default.yml /app/

# Execute the script
ENTRYPOINT ["python3", "run.py", "--config", "default.yml"]
