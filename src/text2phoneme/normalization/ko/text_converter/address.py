import re
from .number import NumConverter


class AddressConverter:
    digit_to_kor = {
    "0": "공", "1": "일", "2": "이", "3": "삼", "4": "사",
    "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구"
    }

    # 번지

    pattern_address = re.compile(r"(\d+)-(\d+)(?=[^\da-zA-Z]|$)")


    @classmethod
    def convert(cls, text: str) -> str:
        
        def address_to_korean(match: re.Match) -> str:
            num1, num2 = match.groups()
            return f"{NumConverter.process_num(num1)} 다시 {NumConverter.process_num(num2)}"
        
        if "-" not in text or not re.search(r"\d", text):
            return text
        
        text = cls.pattern_address.sub(address_to_korean, text)
    

        return text
    

if __name__ == "__main__":
    converter = AddressConverter()
    print(converter.convert("N3N의 도로명 주소는 서울 강남구 강남대로162길 41-18"))
    print(converter.convert("N3N의 도로명 주소는 서울 강남구 강남대로162길 41-18."))
    print(converter.convert("N3N의 도로명 주소는 서울 강남구 강남대로162길 41-18이야"))
    print(converter.convert("N3N의 지번 주소는 서울 강남구 신사동 524-3"))
    print(converter.convert("N3N의 지번 주소는 서울 강남구 신사동 524-3!"))

