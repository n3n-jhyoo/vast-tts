
import importlib
from .settings import SUPPORTED_LANGUAGES

NORMALIZER_INSTANCES = {}

class Normalizer:
    def __init__(self, language="ko", method="basic"):
        '''
        언어 및 정규화 방법에 따라 정규화를 수행하는 함수

        Parameters:
            text (str): 정규화 할 텍스트
            language (str): 사용할 언어 ("ko", "en" 등)
            method (str): 정규화 방식 ("basic", "custom", "advanced") # custom, advanced,... 나중에 추가할 예정

        Returns:
            str: 정규화된 텍스트
        '''
        
        if language not in SUPPORTED_LANGUAGES:
           raise ValueError()

        
        if method not in SUPPORTED_LANGUAGES[language]["method"]:
            raise ValueError()
            
        try:
            key = f"{language}_{method}"
            if key not in NORMALIZER_INSTANCES:
                module_path = f"text2phoneme.normalization.{language}.{method}"
                module = importlib.import_module(module_path)
                NORMALIZER_INSTANCES[key] = module
        
        except ModuleNotFoundError:
            raise ValueError()
        
        self.converter = NORMALIZER_INSTANCES[key].convert 

    def process(self, text: str, **kwargs) -> str:
        """text normalization"""
        return self.converter(text, **kwargs)