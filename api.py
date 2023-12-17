from fastapi import FastAPI, HTTPException
from  fastapi.responses import JSONResponse
import subprocess
from concurrent.futures import ThreadPoolExecutor
import asyncio
from transformers import pipeline

if __name__ == "__main__":executor = ThreadPoolExecutor()
app = FastAPI()

# Model information
models_info = [
    {"name": "leaf-disease", "pretrained_name": "NonoBru/leaf-disease-classifier"},
    {"name": "fruit-leaf-mushrooms", "pretrained_name": "NonoBru/fruit-leaf-mushrooms-classifier"},
    {"name": "health", "pretrained_name": "NonoBru/health-classifier"},
    {"name": "mushroom", "pretrained_name": "NonoBru/health-classifier"},
    {"name": "leaf", "pretrained_name": "NonoBru/leaf-classifier"},
    {"name": "fruit", "pretrained_name": "NonoBru/fruit-classifier-1.0"},
    {"name": "deficiency", "pretrained_name": "NonoBru/deficiency-classifier"},
    {"name": "insect-infection", "pretrained_name": "NonoBru/insect-infection-classifier"},
    {"name": "fruit-disease", "pretrained_name": "NonoBru/fruit-disease-classifier"},
]

# Load models
loaded_models = []
for model_info in models_info:
    print(f"Loading model for {model_info['name']}...")
    model = pipeline("image-classification", model=model_info['pretrained_name'])
    loaded_models.append({"name": model_info['name'], "model": model})
    print(f"Model for {model_info['name']} loaded successfully.\n")

async def classify_image(model, image_bytes):
    print("Classifying image...")
    result = model(image_bytes)
    print("Image classified successfully.")
    return result



# Endpoint for all image classification
@app.post('/classify/{model_name}')
async def classify(model_name: str, data: dict):
    print(f"Received request for model: {model_name}")
    
    if model_name not in [m['name'] for m in loaded_models]:
        raise HTTPException(status_code=404, detail="Model not found")

    image_base64 = data.get("base64Image")
    print(image_base64)
    if not image_base64:
        raise HTTPException(status_code=400, detail="Missing base64Image in JSON payload.")

    model = next(item for item in loaded_models if item['name'] == model_name)['model']
    print("Classifying image using model...")
    result = await asyncio.to_thread(classify_image, model, image_base64)
    awaited_result = await result
    print("Image classified successfully.", awaited_result)
    return JSONResponse(content=awaited_result)

if __name__ == "__main__":
    # Include the uvicorn command
    subprocess.run(["uvicorn", "api:app", "--reload"])