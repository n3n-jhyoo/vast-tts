from importlib.resources import files
import csv
import re

class CustomConverter:

    special_cases = {}
    pattern_special = None

    @classmethod
    def load_cases(cls):
        if cls.special_cases:
            return  # 이미 로딩된 경우 생략

        path = files("text2phoneme.normalization.ko.data").joinpath("sorted_dict.csv")
        with path.open(encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) >= 2:
                    key = row[0].strip().lower()
                    value = row[1].strip()
                    cls.special_cases[key] = value

        # 정규식 패턴 생성
        escaped_keys = map(re.escape, cls.special_cases.keys())
        pattern = r"(?<![A-Za-z0-9])(" + "|".join(escaped_keys) + r")(?=[가-힣\s.,!?]|$)"
        cls.pattern_special = re.compile(pattern, flags=re.IGNORECASE)
    
    @classmethod
    def convert(cls, text: str) -> str:
        cls.load_cases()

        def replace(match: re.Match) -> str:
            matched = match.group(1)
            return cls.special_cases.get(matched.lower(), matched)  # fallback

        return cls.pattern_special.sub(replace, text)


if __name__ == "__main__":
    converter = CustomConverter()
    print(converter.convert("나는 RTX3080을 사서 집으로 갈거임"))
    print(converter.convert("나는 LTE 와이파이 뭐가더 조아?"))