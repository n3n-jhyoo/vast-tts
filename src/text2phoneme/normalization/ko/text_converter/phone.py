import re

class PhoneConverter:
    digit_to_kor = {
    "0": "공", "1": "일", "2": "이", "3": "삼", "4": "사",
    "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구"
    }

    # area_codes 
    area_codes = {
        "02", "031", "032", "033", "041", "042", "043", "044", # 서울, 경기도, 인청광역시, 강원특별자치도, 충청남도, 대전 광역시, 충청북도, 세종특별자치시
        "051", "052", "053", "054", "055", "061", "062", "063", "064" # 부산 광역시, 울산 광역시, 대구광역시, 경상북도, 경상남도, 전라남도, 광주광역시, 전북특별자치도, 제주특별자치도
    }
     
    # 지역 번호
    pattern_area_phone = re.compile(
        r"\b(" + "|".join(sorted(area_codes, key=len, reverse=True)) + r")-\d{3,4}-\d{4}(?=[^\da-zA-Z]|$)"
    )

    # 휴대폰 번호
    pattern_mobile = re.compile(r"\b010-\d{4}-\d{4}(?=[^\da-zA-Z]|$)")
    
    # 대표 번호
    pattern_representative = re.compile(r"\b(15\d{2}|16\d{2}|18\d{2})-\d{4}(?=[^\da-zA-Z]|$)")

    # 안심 번호
    pattern_safe_number = re.compile(r"\b050\d-\d{3,4}-\d{4}(?=[^\da-zA-Z]|$)")


    @classmethod
    def convert(cls, text: str) -> str:
        

        def phone_to_korean(match: re.Match) -> str:

            return "".join(cls.digit_to_kor.get(d, " ") for d in match.group())
        
        if "-" not in text or not re.search(r"\d", text):
            return text
        
        text = cls.pattern_area_phone.sub(phone_to_korean, text)
        text = cls.pattern_mobile.sub(phone_to_korean, text)
        text = cls.pattern_representative.sub(phone_to_korean, text)
        text = cls.pattern_safe_number.sub(phone_to_korean, text)

        return text
    

if __name__ == "__main__":
    converter = PhoneConverter()
    print(type(converter.area_codes))
    print(converter.convert("내 전화번호는 02-111-1111이야"))
    print(converter.convert("내 전화번호는 031-1112-1011이야"))
    print(converter.convert("내 전화번호는 010-6293-9541이야"))
    print(converter.convert("피자헛 대표 번호는 1588-5588이야"))
    print(converter.convert("신사역 샐러디의 번호는 0507-1342-7895입니다."))

