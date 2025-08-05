import re

# 숫자를 한글로 변환하는 클래스

class NumConverter:

    native_bound_nouns = list("군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통 명".split()) # 번 -> 제외
    
    surfix_measure_units = {
    "kg": "킬로그램",
    "g": "그램",
    "km": "킬로미터",
    "m": "미터",
    "cm": "센티미터",
    "mm": "밀리미터",
    "ml": "밀리리터",
    "L": "리터",
    "%" : "퍼센트",
    "°C" : "도씨",
    "°" : "도"
    }
    
    prefix_measure_units = {
        "$": "달러",
        "₩": "원",
        "€" : "유로",
        "¥" : "옌",
        "£" : "파운드",
    }
    
    scale_suffixes = ["만", "억", "조", "경"]

    native_units = {
        "1": "한", "2": "두", "3": "세", "4": "네", "5": "다섯",
        "6": "여섯", "7": "일곱", "8": "여덟", "9": "아홉"
    }
    
    native_tens = {
        "1": "열", "2": "스물", "3": "서른", "4": "마흔", "5": "쉰",
        "6": "예순", "7": "일흔", "8": "여든", "9": "아흔"
    }
    
    sino_units = {
        "1": "일", "2": "이", "3": "삼", "4": "사", "5": "오",
        "6": "육", "7": "칠", "8": "팔", "9": "구"
    }

    pattern_alphabet = re.compile(r"[A-Za-z]")
    
    @classmethod
    def process_num(cls, num: str, sino=True) -> str:
        """숫자를 한글로 변환"""
        if num == "0":
            return "영"

        result = ""
        chunk_num = [num[max(0, i - 4):i] for i in range(len(num), 0, -4)]

        for depth, chunk in enumerate(chunk_num):
            tmp = []
            thousand, hundred, ten, rest = chunk[-4:-3], chunk[-3:-2], chunk[-2:-1], chunk[-1:]

            use_sino = sino or (len(num) > 2)

            if thousand and int(thousand) != 0:
                if int(thousand) != 1:
                    tmp.append(cls.sino_units[thousand])
                tmp.append("천")

            if hundred and int(hundred) != 0:
                if int(hundred) != 1:
                    tmp.append(cls.sino_units[hundred])
                tmp.append("백")

            if ten and int(ten) != 0:
                if use_sino:
                    if int(ten) != 1:
                        tmp.append(cls.sino_units[ten])
                    tmp.append("십")
                else:
                    tmp.append(cls.native_tens[ten])

            if rest and rest[-1] != "0":
                if use_sino:
                    tmp.append(cls.sino_units[rest[-1]])
                else:
                    tmp.append(cls.native_units[rest[-1]])

            if depth > 0 and tmp:
                if len(tmp) == 1 and tmp[-1] == '일' and cls.scale_suffixes[depth - 1] == '만':
                    tmp.pop(-1)
                tmp.append(cls.scale_suffixes[depth - 1])

            result = "".join(tmp) + " " + result

        return result.strip()

    @classmethod
    def process_decimal(cls, num: str) -> str:
        """소수점 숫자를 한글로 변환"""
        dec_units = {str(i): num for i, num in enumerate(["영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"])}
        int_part, dec_part = num.split(".")
        return cls.process_num(int_part) + " 점 " + "".join(dec_units[d] for d in dec_part)

    @classmethod
    def convert(cls, text: str) -> str:
        """숫자를 한글로 변환하는 함수"""

        def replace_match_with_surfix_measure(match):
            sym, num, unit = match.groups()
            num_korean = cls.process_decimal(num) if "." in num else cls.process_num(num, sino=True)
            # return f"{num_korean} {cls.surfix_measure_units[unit]}" if sym != "-" else f"마이너스 {num_korean} {cls.surfix_measure_units[unit]}"
            return f"{num_korean}{cls.surfix_measure_units[unit]}" if sym != "-" else f"마이너스 {num_korean}{cls.surfix_measure_units[unit]}"

        def replace_match_with_prefix_measure(match):
            sym, unit, num = match.groups()
            num_korean = cls.process_decimal(num) if "." in num else cls.process_num(num, sino=True)
            # return f"{num_korean} {cls.prefix_measure_units[unit]}" if sym != "-" else f"마이너스 {num_korean} {cls.prefix_measure_units[unit]}"
            return f"{num_korean}{cls.prefix_measure_units[unit]}" if sym != "-" else f"마이너스 {num_korean}{cls.prefix_measure_units[unit]}"
    
        def replace_match_with_kor(match):
            num, noun = match.groups()
            # sino = noun not in self.native_bound_nouns
            
            # ✅ 숫자 앞 문자가 영어라면 변환하지 않음
            before = match.string[match.start(1) - 1] if match.start(1) > 0 else ''

            if cls.pattern_alphabet.match(before):
                return match.group(0)
            
            # if before.isalpha():
            #     return match.group(0)
            
            noun_= noun
            while noun_ and noun_ not in cls.native_bound_nouns:
                noun_ = noun_[:-1]  # 뒤에서 한 글자씩 제거하면서 찾기
            sino = noun_ not in cls.native_bound_nouns
            num_korean = cls.process_decimal(num) if "." in num else cls.process_num(num, sino=sino)
            # return f"{num_korean} {noun}"
            return f"{num_korean}{noun}"

        
        def replace_match_number(match):
            num = match.group()
            s = match.string
            start = match.start()
            end = match.end()

            before = s[start - 1] if start > 0 else ''
            after = s[end] if end < len(s) else ''

            # 🚫 숫자 앞뒤에 영어가 있으면 변환하지 않음
            # if before.isalpha() or after.isalpha():
            #     return match.group(0)
            if cls.pattern_alphabet.match(before) or cls.pattern_alphabet.match(after):
                return match.group(0)

            return cls.process_decimal(num) if "." in num else cls.process_num(num, sino=True)

        if not re.search(r'\d', text):
            return text

        # 숫자 앞, 위에 있는 쉼표 제거
        text = re.sub(r"(?<=\d),(?=\d)", "", text)
        
        # 숫자 + 측정 단위 
        text = re.sub(r"([-+]?)(\d+\.?\d*)(" + "|".join(cls.surfix_measure_units.keys()) + r")(?![a-zA-Z0-9_])", replace_match_with_surfix_measure, text)
        # 측정 단위 + 숫자
        # "$300USD"
        text = re.sub(r"([-+]?)(" + "|".join(map(re.escape, cls.prefix_measure_units.keys())) + r")(\d+\.?\d*)(?![a-zA-Z])", replace_match_with_prefix_measure, text)        
        # 숫자 + 한글
        text = re.sub(r"(\d+\.?\d*)([가-힣]+)", replace_match_with_kor, text)
        # 숫자만 있는 경우
        text = re.sub(r"\d+\.?\d*", replace_match_number, text)
 
        return text




if __name__ == "__main__":
    converter = NumConverter()
    print(converter.convert("$400"))
    print(converter.convert("내가 받는 용돈은 ₩500이야"))
    print(converter.convert("내가 받는 용돈은 ㄱ₩500이야"))
    print(converter.convert("내 생일을 3월 29일이야! 1993년에 태어났지!"))
    print(converter.convert("12,000초야!"))
    print(converter.convert("12,000개야!"))
    print(converter.convert("1개야"))
    print(converter.convert("14.0203초야"))
    print(converter.convert("1234567890개야!"))
    print(converter.convert("1234567890!! 2485"))
    print(converter.convert("나 RTX3080이거 사러갈거야. 쿠폰 사용할 수 있는데 34E3A야! 25cm지! 3080RTX이거 바꾸지마. E39AK48, 3E8A3, GPT4 사용해본적있니?"))

    # for i in range(1, 100):
    #     print(converter.convert(f"{i}개야"))
    #     print(converter.convert(f"{i}개"))

    #     print(converter.convert(f"{i}초"))
    #     print()



