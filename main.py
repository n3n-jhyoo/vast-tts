import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, src_path)

import gdown

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Depends
import uvicorn

from src.model import TTS

from pydantic import BaseModel
import base64


class PredictRequest(BaseModel):
    text: str
    voice_name: str

class PredictResponse(BaseModel):
    audio: str


app = FastAPI()

def download_model():
    file_id = "1LC2-fgl4uxpjM1aFq9YqGA6QJWj00MNN"
    url = f"https://drive.google.com/uc?id={file_id}"

    output_path = "checkpoints/model.pth"  # 저장할 파일명 원하는 걸로
    gdown.download(url, output_path, quiet=False)


@lru_cache()
def model_load():
    print(" -- TTS Model Loading ---")
    model = TTS(model_path="checkpoints/model.pth", config_path="checkpoints/config.yml")  
    return model

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, model=Depends(model_load)):

    try:
        wav = model.predict(request.text, request.voice_name)
        wav_bytes = wav.tobytes()
        b64_audio = base64.b64encode(wav_bytes).decode("utf-8")

        return {"audio" : b64_audio}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
     
@app.get("/healthcheck")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    download_model()
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=False)