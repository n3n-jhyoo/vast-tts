import os
import re
import sys

import nltk
current_dir = os.path.dirname(__file__)
nltk_data_dir = os.path.join(current_dir, 'resources', 'nltk_data')
nltk.data.path.append(str(nltk_data_dir))


print("NLTK data path:", nltk.data.path)

try:
    path = nltk.data.find('corpora/cmudict')
    print("Found cmudict at:", path)
    from nltk.corpus import cmudict 

except LookupError:
    print(" cmudict not found in any local NLTK data path")

# from  g2p_en import G2p
from .utils import ARPAbet_to_kor


class EnglishConverter:
    cmu_dict = None
    # g2p = None

    pattern_cmu = re.compile(r"[A-Za-z]+", re.IGNORECASE)
    pattern_cmu = re.compile(r"[A-Za-z]+", re.IGNORECASE)

    pattern_digit = re.compile(r"\d")

    @classmethod
    def ensure_initialized(cls):
        if cls.cmu_dict is None:
            cls.cmu_dict = cmudict.dict()
        # if cls.g2p is None:
        #     # cls.g2p = G2p()
        #     cls.g2p.cmudict = cls.cmu_dict
 
    @classmethod
    def convert(cls, text: str) -> str:
        
        def replace_eng(match):
            word = match.group(0)
            s = match.string
            start = match.start()
            end = match.end()
            before = s[start - 1] if start > 0 else ''
            after = s[end] if end < len(s) else ''
            
            # 앞 뒤가 숫자이면 변환하지 않음
            if cls.pattern_digit.match(before) or cls.pattern_digit.match(after):
                return word
            
            # cmu dict에 존재하는 경우
            lower_word = word.lower()
            if lower_word in cls.cmu_dict:
                phoneme = cls.cmu_dict[lower_word][0]  # 첫 번째 발음 사용
                return ARPAbet_to_kor(phoneme)
            
            # g2p 인데 cmudict 내용 관련으로 일단 보류
            # try:
            #     phoneme_list = cls.g2p(lower_word)
            #     return "".join(ARPAbet_to_kor(phoneme_list))
            # except Exception as e:
            #     print(f"[G2P 변환 오류] {word}: {e}")
            #     return word

        if not re.search(r'[A-Za-z]', text):
            return text
        
        cls.ensure_initialized()
        
        # 한 번의 sub로 CMU → G2P fallback 적용
        return cls.pattern_cmu.sub(lambda match: replace_eng(match), text)
    
if __name__ == "__main__":
    converter = EnglishConverter()
    print(converter.convert("내 game은 over됨!ㅋㅋ Happy!! first "))
    print(converter.convert("I love apple and GPU, CPU deep learning 아휴 strong hehe이"))