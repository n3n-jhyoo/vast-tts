import re
from jamo import h2j, j2h

###### ARPAbet 발음 기호를 한국어 발음으로 바꾸는데 활용하는 코드
short_vowels = ("AE", "AH", "AX", "EH", "IH", "IX", "UH")
vowels = "AEIOUY"
consonants = "BCDFGHJKLMNPQRSTVWXZ"
syllable_final_or_consonants = "$BCDFGHJKLMNPQRSTVWXZ"



def adjust(phoneme: str) -> str:
    '''Modify phoneme so that it fits our processes'''
    phoneme = " " + " ".join(phoneme) + " $"
    phoneme = re.sub("\d", "", phoneme)
    phoneme = phoneme.replace(" T S ", " TS ")
    phoneme = phoneme.replace(" D Z ", " DZ ")
    phoneme = phoneme.replace(" AW ER ", " AWER ")
    phoneme = phoneme.replace(" IH R $", " IH ER ")
    phoneme = phoneme.replace(" EH R $", " EH ER ")
    phoneme = phoneme.replace(" $", "")

    return phoneme.strip("$ ").split()

def to_choseong(phoneme):
    '''phoneme to choseong or onset'''
    d = \
        {'G': 'ᄀ', 'N': 'ᄂ', 'D': 'ᄃ', 'DH': 'ᄃ', 'L': 'ᄅ', 'R': 'ᄅ','M': 'ᄆ', 
        'B': 'ᄇ', 'V': 'ᄇ', 'S': 'ᄉ', 'SH': 'ᄉ', 'TH': 'ᄉ', 'NG': 'ᄋ', 'DZ': 'ᄌ',
        'JH': 'ᄌ', 'Z': 'ᄌ',  'ZH': 'ᄌ', 'CH': 'ᄎ', 'TS': 'ᄎ','K': 'ᄏ', 'F': 'ᄑ', 'P': 'ᄑ',
        'T': 'ᄐ',  'HH': 'ᄒ',  'W': 'W', 'Y': 'Y', }

    return d.get(phoneme, phoneme)



def to_jungseong(phoneme):
    '''phoneme to jungseong or vowel'''
    d = \
        {'AA': 'ᅡ','AE': 'ᅢ','AH': 'ᅥ','AO': 'ᅩ','AW': 'ᅡ우','AWER': "ᅡ워",
        'AY': 'ᅡ이', 'EH': 'ᅦ', 'ER': 'ᅥ', 'EY': 'ᅦ이', 'IH': 'ᅵ', 'IY': 'ᅵ',
        'OW': 'ᅩ', 'OY': 'ᅩ이', 'UH': 'ᅮ', 'UW': 'ᅮ'}
    
    return d.get(phoneme, phoneme)


def to_jongseong(phoneme):
    '''phoneme to jongseong or coda'''
    d = \
        {'B': 'ᆸ', 'CH': 'ᆾ', 'D': 'ᆮ', 'DH': 'ᆮ', 'F': 'ᇁ', 'G': 'ᆨ',
        'HH': 'ᇂ',  'JH': 'ᆽ', 'K': 'ᆨ', 'L': 'ᆯ', 'M': 'ᆷ', 'N': 'ᆫ',
        'NG': 'ᆼ',' P': 'ᆸ', 'R': 'ᆯ',  'S': 'ᆺ', 'SH': 'ᆺ', 'T': 'ᆺ',
        'TH': 'ᆺ',  'V': 'ᆸ', 'W': 'ᆼ', 'Y': 'ᆼ', 'Z': 'ᆽ', 'ZH': 'ᆽ'}

    return d.get(phoneme, phoneme)


def process(p, p_prev, p_next, p_next2):
    """자음 처리 (유성/무성 파열음, 마찰음, 파찰음, 비음, 유음)"""
    ret = ""

    # 무성 파열음 [p], [t], [k]
    if p in "PTK":
        if p_prev[:2] in short_vowels and p_next == "$":
            ret += to_jongseong(p)
        elif p_prev[:2] in short_vowels and p_next[0] not in "AEIOULRMN":
            ret += to_jongseong(p)
        elif p_next[0] in "$BCDFGHJKLMNPQRSTVWXYZ":
            ret += to_choseong(p)
            ret += "ᅳ"
        else:
            ret += to_choseong(p)

    # 유성 파열음 [b], [d], [g]
    elif p in "BDG":
        ret += to_choseong(p)
        if p_next[0] in syllable_final_or_consonants:
            ret += "ᅳ"

    # 마찰음 [s], [z], [f], [v], [θ], [ð], [ʃ], [ʒ]
    elif p in ("S", "Z", "F", "V", "TH", "DH", "SH", "ZH"):
        ret += to_choseong(p)

        if p in ("S", "Z", "F", "V", "TH", "DH"):
            if p_next[0] in syllable_final_or_consonants:
                ret += "ᅳ"
        elif p == "SH":
            if p_next[0] in "$":
                ret += "ᅵ"
            elif p_next[0] in consonants:
                ret += "ᅲ"
            else:
                ret += "Y"
        elif p == "ZH":
            if p_next[0] in syllable_final_or_consonants:
                ret += "ᅵ"

    # 파찰음 [ʦ], [ʣ], [ʧ], [ʤ]
    elif p in ("TS", "DZ", "CH", "JH"):
        ret += to_choseong(p)
        if p_next[0] in syllable_final_or_consonants:
            ret += "ᅳ" if p in ("TS", "DZ") else "ᅵ"

    # 비음 [m], [n], [ŋ]
    elif p in ("M", "N", "NG"):
        if p in "MN" and p_next[0] in vowels:
            ret += to_choseong(p)
        else:
            ret += to_jongseong(p)

    # 유음 [l]
    elif p == "L":
        if p_prev == "^":
            ret += to_choseong(p)
        elif p_next[0] in "$BCDFGHJKLPQRSTVWXZ":
            ret += to_jongseong(p)
        elif p_prev in "MN":
            ret += to_choseong(p)
        elif p_next[0] in vowels:
            ret += "ᆯᄅ"
        elif p_next in "MN" and p_next2[0] not in vowels:
            ret += "ᆯ르"

        # custom
    elif p == "ER":
        if p_prev[0] in vowels:
            ret += "ᄋ"
        ret += to_jungseong(p)
        if p_next[0] in vowels:
            ret += "ᄅ"
    elif p == "R":
        if p_next[0] in vowels:
            ret += to_choseong(p)

    # 8항. 중모음1) ([ai], [au], [ei], [ɔi], [ou], [auə])
    # 중모음은 각 단모음의 음가를 살려서 적되, [ou]는 '오'로, [auə]는 '아워'로 적는다.
    elif p[0] in "AEIOU":
        ret += to_jungseong(p)

    else:
        ret += to_choseong(p)
    return ret


def reconstruct(string):
    '''Some postprocessing rules'''
    pairs = [("그W", "ᄀW"), ("흐W", "ᄒW"),  ("크W", "ᄏW"), ("ᄂYᅥ", "니어"),
            ("ᄃYᅥ", "디어"),  ("ᄅYᅥ", "리어"),  ("Yᅵ", "ᅵ"),  ("Yᅡ", "ᅣ"),
            ("Yᅢ", "ᅤ"), ("Yᅥ", "ᅧ"),("Yᅦ", "ᅨ"), ("Yᅩ", "ᅭ"),("Yᅮ", "ᅲ"),
            ("Wᅡ", "ᅪ"), ("Wᅢ", "ᅫ"), ("Wᅥ", "ᅯ"), ("Wᅩ", "ᅯ"), ("Wᅮ", "ᅮ"),
            ("Wᅦ", "ᅰ"),  ("Wᅵ", "ᅱ"), ("ᅳᅵ", "ᅴ"), ("Y", "ᅵ"), ("W", "ᅮ")]
    
            
    for str1, str2 in pairs:
        string = string.replace(str1, str2)

    return string

def compose(letters):
    # insert placeholder
    # insert placeholder
    letters = re.sub("(^|[^\u1100-\u1112])([\u1161-\u1175])", r"\1ᄋ\2", letters)

    string = letters # assembled characters
    # c+v+c
    syls = set(re.findall("[\u1100-\u1112][\u1161-\u1175][\u11A8-\u11C2]", string))
    for syl in syls:
        string = string.replace(syl, j2h(*syl))

    # c+v
    syls = set(re.findall("[\u1100-\u1112][\u1161-\u1175]", string))
    for syl in syls:
        string = string.replace(syl, j2h(*syl))
    return string

def ARPAbet_to_kor(phoneme:str) -> str:
    phoneme = adjust(phoneme)
    ret = ""

    for i in range(len(phoneme)):
        p = phoneme[i]
        p_prev = phoneme[i - 1] if i > 0 else "^"
        p_next = phoneme[i + 1] if i < len(phoneme) - 1 else "$"
        p_next2 = phoneme[i + 1] if i < len(phoneme) - 2 else "$"
        ret += process(p, p_prev, p_next, p_next2)


    ret = reconstruct(ret)
    ret = compose(ret)
    ret = re.sub("[\u1100-\u11FF]", "", ret) # remove hangul jamo
    return ret  