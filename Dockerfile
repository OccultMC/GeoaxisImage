FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Install vastai CLI for self-destruct
RUN pip install --no-cache-dir vastai

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY pipeline.py r2_storage.py entrypoint.sh ./
COPY gsvpd/ ./gsvpd/

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
