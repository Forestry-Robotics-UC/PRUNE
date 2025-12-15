FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY entfac_fusion_core /app/entfac_fusion_core
COPY tests /app/tests

# Run core unit tests; ROS is not required for numpy-only tests.
CMD ["pytest", "-q"]
