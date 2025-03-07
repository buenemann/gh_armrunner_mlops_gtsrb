# Use the official PyTorch image as the base image
# FROM armswdev/pytorch-arm-neoverse:r24.07-torch-2.3.0-onednn-acl
FROM python:latest

# Set the working directory
WORKDIR /app

# Copy the necessary files
COPY models/ /app/models/
COPY scripts/ /app/scripts/
COPY test-img.png /app/

# Install any additional dependencies
RUN pip install --no-cache-dir torch torchvision fastapi uvicorn python-multipart

# Expose the port the app will run on
EXPOSE 8000

# Specify the command to run the model server
CMD ["uvicorn", "scripts.serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
