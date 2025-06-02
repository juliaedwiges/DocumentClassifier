from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from torchvision import transforms, models
import torch.nn as nn
import uvicorn
import os

app = FastAPI()

# Defina as classes conforme seu dataset
classes = ['CNH_Aberta', 'CNH_Frente', 'CNH_Verso', 'CPF_Frente', 'CPF_Verso', 'RG_Aberto', 'RG_Frente', 'RG_Verso']

# Transformações da imagem para o modelo
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Caminho para o modelo
model_path = os.path.join('model', 'document_classifier.pth')

# Reconstrua a arquitetura do modelo
def get_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))  # 8 classes
    return model

# Inicialize e carregue os pesos
device = torch.device("cpu")
model = get_model()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        predicted_class = classes[predicted.item()]
    return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence": round(confidence.item(), 4)
    })

