# Use the official PyTorch image as the base image
FROM pytorch/pytorch:2.4.0-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the necessary files
COPY models/ /app/models/
COPY scripts/ /app/scripts/

# Install any additional dependencies
RUN pip install --no-cache-dir torch torchvision fastapi uvicorn

# Expose the port the app will run on
EXPOSE 8000

# Specify the command to run the model server
CMD ["uvicorn", "scripts.serve_model:app", "--host", "0.0.0.0", "--port", "8000"]