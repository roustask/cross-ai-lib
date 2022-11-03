import re
import demoji
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker


def clean_emojis(text: str) -> str:
    for emoji in demoji.findall(text):
        text = text.replace(emoji, " ")

    regrex_pattern = re.compile(pattern="["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    text = regrex_pattern.sub(r'', text)

    text = remove_redundant_spaces(text)
    return text


def remove_redundant_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def symbols_cleaner(
        text: str,
        symbols: str = r"[`~!@#$%^&*()\-+=\[\]{\}/?><,\'\":;|»«§°·¦ʼ¬£€©΄”¨•“.’‘´\\…]"
) -> str:
    text = re.sub(symbols, " ", text)
    # Removing multiple spaces
    text = remove_redundant_spaces(text)
    return text


def stopwords_remover(text: str) -> str:
    stopwords_set = set(stopwords.words('english'))
    non_stopwords = [
        word for word in text.split() if word.lower() not in stopwords_set
    ]
    return " ".join(non_stopwords)


def spell_check(text: str) -> str:
    spell = SpellChecker()
    misspelled_words = spell.unknown(text.split())
    for word in misspelled_words:
        if spell.correction(word) is not None:
            text = text.replace(word, spell.correction(word))
    return text


def lemmatize(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [
        lemmatizer.lemmatize(word) for word in text.split()
    ]
    return " ".join(lemmatized_text)


def clean_numbers(text: str) -> str:
    text = re.sub(r"\d+", " ", text)
    text = remove_redundant_spaces(text)
    return text


def clean_urls(text: str) -> str:
    text = re.sub(r"http\S+", " ", text)
    text = remove_redundant_spaces(text)
    return text


def clean_mentions(text: str) -> str:
    text = re.sub(r"@\S+", " ", text)
    text = remove_redundant_spaces(text)
    return text


def clean_hashtags(text: str) -> str:
    text = re.sub(r"#\S+", " ", text)
    text = remove_redundant_spaces(text)
    return text


def remove_repeated_characters(text: str) -> str:
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = remove_redundant_spaces(text)
    return text


def replace_contractions(text: str) -> str:
    contractions = pd.read_csv(
        'datasets/contractions.zip', index_col='Contraction', compression='zip')
    contractions.index = contractions.index.str.lower()
    contractions.Meaning = contractions.Meaning.str.lower()
    contractions_dict = contractions.to_dict()['Meaning']
    for contraction, replacement in contractions_dict.items():
        text = text.replace(contraction, replacement)
    text = remove_redundant_spaces(text)
    return text