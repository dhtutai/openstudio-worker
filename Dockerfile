FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py .
ENV HF_HOME=/runpod-volume/huggingface
CMD ["python", "-u", "handler.py"]
