import re

from text2phoneme.normalization.ko.text_converter.number import NumConverter
from text2phoneme.normalization.ko.text_converter.custom import CustomConverter
from text2phoneme.normalization.ko.text_converter.english import EnglishConverter  
from text2phoneme.normalization.ko.text_converter.mix import MixConverter
from text2phoneme.normalization.ko.text_converter.phone import PhoneConverter
from text2phoneme.normalization.ko.text_converter.address import AddressConverter



def convert(text: str) -> str:
        
    if not re.search(r'\d', text) and not re.search(r'[A-Za-z]',text):
        return text

    text = CustomConverter.convert(text)
    text = PhoneConverter.convert(text)
    text = AddressConverter.convert(text)
    text = NumConverter.convert(text)
    text = EnglishConverter.convert(text)
    text = MixConverter.convert(text)

    return text

if __name__ == "__main__":
    text = "나는 RTX3080은 크기가 얼마나되는거지? 24cm정도인가?"
    text = "체온이 -2°C까지 떨어졌습니다."
    text = convert(text)
    print(text)