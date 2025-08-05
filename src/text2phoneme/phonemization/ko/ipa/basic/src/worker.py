# the engine that does the hard lifting.
# convert() is the entry point for converting Korean orthography into transcription
import os
import regex as re
from base64 import b64decode
from typing import Union
from pathlib import Path

# from src.classes import ConversionTable, Word
from .classes import ConversionTable, Word
# import src.rules as rules
from . import rules

from transformers import AutoTokenizer

model_dir = Path(__file__).parent / 'resources' / 'bert_kor_base'
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

# for stts2 
import string
allowed_punctuation = r'가-힣;:,.!?¡¿—…"«»“” '


def transcription_convention(convention: str):
    # supported transcription conventions: ipa, yale, park
    convention = convention.lower()
    if convention not in ['ipa', 'yale', 'park']:
        raise ValueError(f"Your input {convention} is not supported.")
    return ConversionTable(convention)


def sanitize(word: str) -> str:
    """
    converts all hanja 漢字 letters to hangul
    and also remove any space in the middle of the word
    """
    if len(word) < 1:  # if empty input, no sanitize
        return word

    word = word.replace(' ', '')

    hanja_idx = [match.start() for match in re.finditer(r'\p{Han}', word)]
    if len(hanja_idx) == 0:  # if no hanja, no sanitize
        return word

    # from src.hanja_tools import hanja_cleaner  # import hanja_cleaner only when needed
    from .hanja_tools import hanja_cleaner  # import hanja_cleaner only when needed
    
    r = hanja_cleaner(word, hanja_idx)
    return r


def convert(hangul: str,
            rules_to_apply: str = 'pastcnhovr',
            convention: str = 'ipa',
            sep: str = '') -> str:
    
    # the main function for IPA conversion

    if len(hangul) < 1:  # if no content, then return no content
        return ""

    # prepare
    rules_to_apply = rules_to_apply.lower()
    CT_convention = transcription_convention(convention)
    # hangul = sanitize(hangul) # hanja to hangul
    word = Word(hangul=hangul)

    # resolve word-final consonant clusters right off the bat
    rules.simplify_coda(word) # 겹받침(받침 클러스터)을 단순화

    # apply rules
    word = rules.apply_rules(word, rules_to_apply)

    # convert to IPA or Yale
    transcribed = rules.transcribe(word.jamo, CT_convention)

    # apply phonetic rules
    if CT_convention.name == 'ipa':
        transcribed = rules.apply_phonetics(transcribed, rules_to_apply)

    return sep.join(transcribed)



def convert_sentence(sentence: str,
            rules_to_apply: str = 'pastcnhovr',
            convention: str = 'ipa',
            sep: str = '',
            use_tkn: bool = False,
            use_space: bool = True,
            **kwargs
            ):

    use_tkn = kwargs.get("use_tkn", use_tkn)
    use_space = kwargs.get("use_space", use_space)
    sep = kwargs.get("sep", sep)

    full_transcribed = ""
    
    if use_tkn: # tokenizer를 사용하여 문장을 나누는 경우
        tokenized = tokenizer.tokenize(sentence)

        phs = []
        ph_groups = []
        for t in tokenized:
            if not t.startswith("#"):
                ph_groups.append([t])
            else:
                ph_groups[-1].append(t.replace("#", ""))

        split_result = ["".join(group) for group in ph_groups]
        
        for segment in split_result:
            if re.match(r'[가-힣]+', segment):
                transcribed = convert(hangul=segment,
                                        rules_to_apply=rules_to_apply,
                                        convention=convention,
                                        sep=sep)
                full_transcribed+=transcribed + " "
            else:
                full_transcribed += segment + " "

        full_transcribed = full_transcribed.strip()
        return  full_transcribed


    else: # tokenizer를 사용하지 않고 한글과 문자를 분리하는 경우 
        split_result = re.findall(r'[가-힣]+|[^\s가-힣]', sentence)
        if use_space:
            for segment in split_result:
                if re.match(r'[가-힣]+', segment):
                    transcribed = convert(hangul=segment,
                                            rules_to_apply=rules_to_apply,
                                            convention=convention,
                                            sep=sep)
                    full_transcribed+=transcribed + " "
                else:
                    full_transcribed += segment + " "

            full_transcribed = full_transcribed.strip()
            
            return  full_transcribed
        else:
            for segment in split_result:
                if re.match(r'[가-힣]+', segment):
                    transcribed = convert(hangul=segment,
                                            rules_to_apply=rules_to_apply,
                                            convention=convention,
                                            sep=sep)
                    full_transcribed+=transcribed + " "
                else:
                    full_transcribed = full_transcribed.rstrip() + segment + " "

            full_transcribed = full_transcribed.strip()
            
            return  full_transcribed      


def convert_many(long_content: str,
                 rules_to_apply: str = 'pastcnhovr',
                 convention: str = 'ipa',
                 sep: str = '') -> Union[int, str]:
    # decode uploaded file and create a wordlist to pass to convert()
    decoded = b64decode(long_content).decode('utf-8')
    decoded = decoded.replace('\r\n', '\n').replace('\r', '\n')  # normalize line endings
    decoded = decoded.replace('\n\n', '')  # remove empty line at the file end

    input_internal_sep = '\t' if '\t' in decoded else ','

    if '\n' in decoded:
        # a vertical wordlist uploaded
        input_lines = decoded.split('\n')
        wordlist = [l.split(input_internal_sep)[1].strip() for l in input_lines if len(l) > 0]
    else:
        # a horizontal wordlist uploaded
        wordlist = decoded.split(input_internal_sep)

    # iterate over wordlist and populate res
    res = ['Orthography\tIPA']
    for word in wordlist:
        converted_r = convert(hangul=word,
                              rules_to_apply=rules_to_apply,
                              convention=convention,
                              sep=sep)
        res.append(f'{word.strip()}\t{converted_r.strip()}')

    return '\n'.join(res)