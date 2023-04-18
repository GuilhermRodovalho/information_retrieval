from collections import Counter
from math import log2
from typing import Dict, Iterable, List


class IdfBuilder:
    terms_quantity: Dict[str, int]
    idf: Dict[str, float]

    def __init__(self, vocabulary: List[str] = []) -> None:
        self.terms_quantity = {}
        self.idf = {}
        self.documents_quantity = 0
        self.vocabulary = vocabulary

    def add_document_terms(self, terms: Iterable[str]) -> None:
        self.documents_quantity += 1
        for term in terms:
            if self.vocabulary and term not in self.vocabulary:
                continue

            if term in self.terms_quantity:
                self.terms_quantity[term] += 1
            else:
                self.terms_quantity[term] = 1

    def calculate_idf(self) -> None:
        for term, quantity in self.terms_quantity.items():
            self.idf[term] = log2(self.documents_quantity / quantity)

    def __getitem__(self, term: str) -> float:
        if term not in self.idf:
            return 0
        return self.idf[term]

    def __str__(self) -> str:
        res = ""
        for term, idf in self.idf.items():
            res += f"{term}: {idf}\n"

        return res


class TfBuilder:
    @staticmethod
    def calculate_tf(words: List[str], vocabulary: List[str]) -> Dict[str, float]:
        """
        Given a list of words and a vocabulary, calculate the tf of each word in the
        document that is in the vocabulary, and 0 for the ones that are only in the
        vocabulary

        :param document_file: a file name
        :param vocabulary: a list of words
        :return: a dict of the type {word: tf}
        """
        words_frequency = Counter(words)

        # tf = {word: 1 + log2(frequency) for word, frequency in words_frequency.items()}

        # calculate tf using the words that are in the vocabulary
        tf = {}
        for word in vocabulary:
            if word in words_frequency:
                tf[word] = 1 + log2(words_frequency[word])
            else:
                tf[word] = 0

        return tf
