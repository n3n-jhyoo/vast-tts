import importlib
from .settings import SUPPORTED_LANGUAGES

# 생성된 인스턴스 저장 캐시 (singleton)
PHONEMIZER_INSTANCES = {}

class Phonemizer:
    
    def __init__(self, language="ko", phonetic="ipa", method="basic"):
        
        
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError()

        if phonetic not in SUPPORTED_LANGUAGES[language]["phonetic"]:
            raise ValueError()

        if method not in SUPPORTED_LANGUAGES[language]["method"][phonetic]:
            raise ValueError()

        try: 
            key = f"{language}_{phonetic}_{method}"
            if key not in PHONEMIZER_INSTANCES:
                module_path = f"text2phoneme.phonemization.{language}.{phonetic}.{method}"
                module = importlib.import_module(module_path)
                PHONEMIZER_INSTANCES[key] = module

        except ModuleNotFoundError:
            raise ValueError(f"{module_path}")

        self.converter = PHONEMIZER_INSTANCES[key].convert


    def process(self, text: str, **kwargs) -> str:
        return self.converter(text, **kwargs)