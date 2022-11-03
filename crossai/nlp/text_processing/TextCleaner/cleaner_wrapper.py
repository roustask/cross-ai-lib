import multiprocessing
import os
import queue
from pathlib import Path
from typing import Dict, List, Optional

import tqdm

from TextCleaner.preprocessing import TextPreprocessing

# # In order to pass the kwargs to the clean_data_pack module, we would have to add kwargs in a list, like list.append(**kwargs).
# # But this raises an Error. So we use an empty global variable, that we assign values to it later on.
# kwargs = {}


def clean_data_pack(
    q,
    lock,
    lowercase,
    lemmatize,
    spellcheck,
    remove_stopwords,
    remove_emojis,
    remove_numbers,
    remove_urls,
    remove_mentions,
    remove_symbols,
    remove_hashtags,
    remove_repeated_characters,
    replace_contractions
):

    # Create a custom progress bar. This progress bar is shared between all processes.
    init_size = q.qsize()
    with lock:
        bar = tqdm.tqdm(desc="Cleaning", total=init_size, leave=False)

    cleaner = TextPreprocessing(
        lowercase=lowercase,
        lemmatize=lemmatize,
        spellcheck=spellcheck,
        remove_stopwords=remove_stopwords,
        remove_emojis=remove_emojis,
        remove_numbers=remove_numbers,
        remove_urls=remove_urls,
        remove_mentions=remove_mentions,
        remove_symbols=remove_symbols,
        remove_hashtags=remove_hashtags,
        remove_repeated_characters=remove_repeated_characters,
        replace_contractions=replace_contractions
    )

    cleaned_data = []

    while not q.empty():
        try:
            utterance_id, utterance = q.get(True, 1.0)
            cleaned_data.append((utterance_id, cleaner.clean_input(utterance)))
            # Update the progress bar.
            # A child process sees the remaining texts inside the shared queue and knows that we have processed (init_size - queue_size) texts so far.
            # So it has to update the progress bar by that number.
            # Example: init_size = 10 and the first time that a child process tries to update the progress bar, the queue has 6 texts left.
            # So the first update must "move" the progress bar by (10 - 6) = 4.
            size = q.qsize()
            with lock:
                bar.update(init_size - size)
                init_size = size
        except queue.Empty:
            pass
    # Close the progress bar and then return.
    with lock:
        bar.close()
    return cleaned_data


class TextCleaner:
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
        n_jobs: int = 1
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
        self.n_jobs = n_jobs
        self.cores = self._get_num_parallel_jobs(n_jobs=n_jobs)
        if self.n_jobs == 1:
            self.cleaner = TextPreprocessing(
                lowercase=lowercase,
                lemmatize=lemmatize,
                spellcheck=spellcheck,
                remove_stopwords=remove_stopwords,
                remove_emojis=remove_emojis,
                remove_numbers=remove_numbers,
                remove_urls=remove_urls,
                remove_mentions=remove_mentions,
                remove_symbols=remove_symbols,
                remove_hashtags=remove_hashtags,
                remove_repeated_characters=remove_repeated_characters,
                replace_contractions=replace_contractions
            )

    def _get_num_parallel_jobs(self, n_jobs: int) -> int:
        if n_jobs < -1 or n_jobs == 0:
            raise ValueError(
                "Please provide a positive number or -1 as the n_jobs.")
        cores = multiprocessing.cpu_count()
        if n_jobs == -1 or n_jobs > cores:
            return cores
        else:
            return n_jobs

    def clean_input(self, data):
        """Cleans the input data.

        Args:
            data (list(str)): A list of strings to be cleaned.

        Returns:
            list(str): A list of cleaned strings.
        """
        if self.n_jobs == 1:
            cleaned_data = self.cleaner.clean_input(data)
        else:
            if isinstance(data, str):
                raise RuntimeError(
                    "n_jobs is not equal to 1 and you are passing a single String. If you want jobs to run in parallel please provide a List of Strings or else change the n_jobs to 1."
                )
            elif isinstance(data, list):
                if not all(isinstance(elem, str) for elem in data):
                    raise TypeError("List must contain only Strings.")

                m = multiprocessing.Manager()
                q = m.Queue()

                texts_idxs = {}
                for text_idx, text in enumerate(data):
                    if text not in texts_idxs:
                        texts_idxs[text] = []
                    texts_idxs[text].append(text_idx)

                data_without_dups = list(texts_idxs)
                for utterance_id, utterance in enumerate(data_without_dups):
                    q.put_nowait((utterance_id, utterance))

                lock = multiprocessing.Manager().Lock()

                pool_args = []
                for _ in range(self.cores):
                    pool_args.append(
                        (
                            q,
                            lock,
                            self.lowercase,
                            self.lemmatize,
                            self.spellcheck,
                            self.remove_stopwords,
                            self.remove_emojis,
                            self.remove_numbers,
                            self.remove_urls,
                            self.remove_mentions,
                            self.remove_symbols,
                            self.remove_hashtags,
                            self.remove_repeated_characters,
                            self.replace_contractions
                        )
                    )

                with multiprocessing.Pool(self.cores) as p:
                    cleaned_data_packs = p.starmap(clean_data_pack, pool_args)

                cleaned_data_without_dups = [
                    item
                    for cleaned_data_pack in cleaned_data_packs
                    for item in cleaned_data_pack
                ]
                cleaned_data_without_dups = sorted(
                    cleaned_data_without_dups, key=lambda k: k[0]
                )
                cleaned_data_without_dups = [
                    cleaned_text for _, cleaned_text in cleaned_data_without_dups
                ]

                cleaned_data = [None] * len(data)
                for text, cleaned_text in zip(
                    data_without_dups, cleaned_data_without_dups
                ):
                    text_idxs = texts_idxs[text]
                    for text_idx in text_idxs:
                        cleaned_data[text_idx] = cleaned_text
            else:
                raise TypeError("Please pass a String or a List of Strings.")
        return cleaned_data
