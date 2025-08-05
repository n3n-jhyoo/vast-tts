import re

# 숫자+영어 또는 영어+숫자 인 경우 변환


class MixConverter:

    alphabet_to_kor = {
        "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프",
        "G": "지", "H": "에이치", "I": "아이", "J": "제이", "K": "케이",
        "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피", "Q": "큐",
        "R": "알", "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유",
        "X": "엑스", "Y": "와이", "Z": "지"
    }
    digit_to_kor = {
        "0" :"공",
        "1": "일", "2": "이", "3": "삼", "4": "사", "5": "오",
        "6": "육", "7": "칠", "8": "팔", "9": "구"
    }
    
    # pattern_mixed = re.compile(r'(?=\w*[A-Za-z])(?=\w*\d)\w+')
    pattern_mixed = re.compile(r'(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+')

    pattern_alphabet = re.compile(r"[A-Za-z]")
    pattern_digit = re.compile(r"[0-9]")
    
    @classmethod
    def convert_token(cls, match: re.Match) -> str:
        word = match.group(0)
        result = []
        for ch in word:
            if cls.pattern_alphabet.match(ch):
                result.append(cls.alphabet_to_kor.get(ch.upper(), ch))
            elif cls.pattern_digit.match(ch): 
                result.append(cls.digit_to_kor[ch])
            else:
                result.append(ch)
        return "".join(result)

    @classmethod
    def convert(cls, text: str) -> str:
        if not re.search(r'\d', text) or not re.search(r'[A-Za-z]',text):
            return text
        
        return cls.pattern_mixed.sub(cls.convert_token, text)




if __name__ == "__main__":
    converter = MixConverter()
    print(converter.convert("$400"))
    print(converter.convert("내가 받는 용돈은 ₩500이야"))
    print(converter.convert("내가 받는 용돈은 ㄱ₩500이야"))
    print(converter.convert("나는 RTX3080과 AI2024 기술이 들어간 E39AK48을 봤어!"))
    print(converter.convert("12,000초야!"))
    print(converter.convert("12,000개야!"))
    print(converter.convert("1개야"))
    print(converter.convert("14.0203초야"))
    print(converter.convert("1234567890개야!"))
    print(converter.convert("1234567890!! 2485"))
    print(converter.convert("나 RTX3080이거 사러갈거야. 쿠폰 사용할 수 있는데 34E3A야! 25cm지! 3080RTX이거 바꾸지마. E39A0K48, 3E8A3, GPT4 사용해본적있니?"))

    # for i in range(1, 100):
    #     print(converter.convert(f"{i}개야"))
    #     print(converter.convert(f"{i}개"))

    #     print(converter.convert(f"{i}초"))
    #     print()



