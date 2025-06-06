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

model = None

def get_model():
    global model
    if model is None:
        print("🔄 Carregando o modelo...")
        model_path = os.path.join(os.getcwd(), 'model', 'document_classifier.pth')
        print(f"📁 Caminho do modelo: {model_path}")
        if not os.path.exists(model_path):
            raise RuntimeError("❌ Modelo não encontrado!")
        model_resnet = models.resnet18(pretrained=False)
        num_ftrs = model_resnet.fc.in_features
        model_resnet.fc = nn.Linear(num_ftrs, len(classes))
        model_resnet.load_state_dict(torch.load(model_path, map_location=device))
        model_resnet.eval()
        model = model_resnet
        print("✅ Modelo carregado com sucesso.")
    return model

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print("📥 Recebendo imagem...")
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print("🧪 Transformando imagem...")
        input_tensor = transform(image).unsqueeze(0)
        print("🤖 Fazendo predição...")
        with torch.no_grad():
            outputs = get_model()(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            predicted_class = classes[predicted.item()]
        print(f"✅ Resultado: {predicted_class} ({confidence.item()})")
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": round(confidence.item(), 4)
        })
    except Exception as e:
        print(f"❌ Erro: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


