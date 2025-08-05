import requests
import base64

import numpy as np
import soundfile as sf

# FastAPI 서버 주소
url = "http://localhost:5001/predict"

# 요청할 텍스트
data = {
    "text": "오늘은 8월 5일 화요일입니다. 오늘 월급이 들어왔어요!",
    "voice_name" : "sample_male"
}

# POST 요청
response = requests.post(url, json=data)

# 응답 확인
if response.status_code == 200:
    b64_audio = response.json()['audio']
    
    wav_bytes = base64.b64decode(b64_audio)
    waveform = np.frombuffer(wav_bytes, dtype=np.float32)
    sf.write("output.wav", waveform, samplerate=24000)
    print("음성 생성 및 저장 완료")    

else:
    print("오류 발생:", response.status_code, response.text)