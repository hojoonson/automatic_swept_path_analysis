FROM tensorflow/tensorflow:2.7.1-gpu

RUN apt-get update && apt-get -y install --no-install-recommends \
    libfreetype6-dev \
    libportmidi-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libgl1-mesa-glx\
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
