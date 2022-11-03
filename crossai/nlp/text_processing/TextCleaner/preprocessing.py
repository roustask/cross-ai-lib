from nltk.corpus import wordnet as wn
from TextCleaner.helpers import *
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm


class TextPreprocessing:
    def __init__(
        self,
        lowercase: bool = True,
        lemmatize: bool = True,
        spellcheck: bool = True,
        remove_stopwords: bool = True,
        remove_emojis: bool = True,
        remove_numbers: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_symbols: bool = True,
        remove_hashtags: bool = True,
        remove_repeated_characters: bool = True,
        replace_contractions: bool = True,
    ) -> None:
        """
        Creates a pipeline for cleaning English text data.

        Parameters
        -------
        `lowercase`: bool
            Whether to return the input texts in lower case.
        `lemmatize`: bool
            reduce a given word to its root word.
        `spellcheck`: bool
            Spell checks words. If a word is misspelled, it will be replaced with the most likely correct spelling.
        `remove_stopwords`: bool
            Whether to remove the stop words before returning the texts.
        `remove_emojis`: bool
            Whether to remove the emojis from the texts.
        `remove_numbers`: bool
            Whether to remove the numbers from the texts.
        `remove_urls`: str
            Whether to remove the urls from the texts.
        `remove_mentions`: float
            Whether to remove the mentions from the texts.
        `remove_symbols`: bool
            Whether to remove symbols from the texts. Uses a regular expression format.
        `remove_hashtags`: bool
            Whether to remove the hashtags from the texts.
        `remove_repeated_characters`: bool
            Whether to remove repeated characters from the texts.
        `replace_contractions`: bool
            Whether to replace contractions from the texts.

        Returns
        `None`
        """
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.spellcheck = spellcheck
        self.remove_stopwords = remove_stopwords
        self.remove_emojis = remove_emojis
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_symbols = remove_symbols
        self.remove_hashtags = remove_hashtags
        self.remove_repeated_characters = remove_repeated_characters
        self.replace_contractions = replace_contractions

    def clean_input(self, data: Union[str, List[str]]):
        if isinstance(data, str):
            cleaned_data = self._clean_single_text(data)
        elif isinstance(data, list):
            if not all(isinstance(elem, str) for elem in data):
                raise TypeError("List must contain only Strings.")
            texts_idxs = {}
            for text_idx, text in enumerate(data):
                if text not in texts_idxs:
                    texts_idxs[text] = []
                texts_idxs[text].append(text_idx)
            cleaned_data = [None] * len(data)
            for text, text_idxs in tqdm(
                texts_idxs.items(), total=len(texts_idxs), desc="Cleaning"
            ):
                cleaned_text = self._clean_single_text(text)
                for text_idx in text_idxs:
                    cleaned_data[text_idx] = cleaned_text
        else:
            raise TypeError("Please pass a String or a List of Strings.")
        return cleaned_data

    def _clean_single_text(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        if self.remove_emojis:
            text = clean_emojis(text)
        if self.remove_numbers:
            text = clean_numbers(text)
        if self.remove_urls:
            text = clean_urls(text)
        if self.remove_mentions:
            text = clean_mentions(text)
        if self.remove_hashtags:
            text = clean_hashtags(text)
        if self.lowercase:
            text = text.lower()
        if self.replace_contractions:
            text = replace_contractions(text)
        if self.remove_symbols:
            text = symbols_cleaner(text)
        if self.remove_repeated_characters:
            text = remove_repeated_characters(text)
        if self.spellcheck:
            text = spell_check(text)
        if self.remove_stopwords:
            text = stopwords_remover(text)
        if self.lemmatize:
            text = lemmatize(text)
        return text