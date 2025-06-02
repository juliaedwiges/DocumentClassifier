from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from torchvision import transforms, models
import torch.nn as nn
import os

app = FastAPI()

classes = ['CNH_Aberta', 'CNH_Frente', 'CNH_Verso', 'CPF_Frente', 'CPF_Verso', 'RG_Aberto', 'RG_Frente', 'RG_Verso']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

device = torch.device("cpu")

@app.on_event("startup")
def load_model():
    global model
    # Caminho absoluto para o modelo na pasta ../model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'document_classifier.pth')
    model_path = os.path.abspath(model_path)
    
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
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
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
