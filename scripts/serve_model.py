import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

# Define the model class (should match the model architecture used during training)
class TrafficSignNet(torch.nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = torch.nn.Linear(64 * 6 * 6, 128)
        self.fc2 = torch.nn.Linear(128, 43) # 43 classes in GTSRB dataset

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Load the trained model
model = TrafficSignNet()
model.load_state_dict(torch.load("/app/models/traffic_sign_net.pth",
map_location=torch.device('cpu')))
model.eval()

# Define image transformations (should match the preprocessing used during training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Initialize FastAPI
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
        # Preprocess the image
        image = transform(image).unsqueeze(0) # Add batch dimension

        # Run the model on the image
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        # Return the prediction
        return {"predicted_class": predicted.item()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


