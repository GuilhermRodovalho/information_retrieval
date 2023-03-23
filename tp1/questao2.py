from typing import List, Dict, Iterable
from utils import read_all_terms_from_file_to_lower

BagOfWords = List[int]


def bag_of_words(vocabulary: List[str], document_words: List[str]) -> BagOfWords:
    """
    Given a vocabulary and a document returns the bag of words of this document

    example:
    ::
        bag = bag_of_words(["eu", "sou" lindo"], ["o guilherme sou eu"])
        assert bag = [1, 1, 0]

    :param vocabulary: should be a vocabulary, without repeted words
    :param document_words: a list of words to be compared to the vocabulary
    :return:
    """
    return [1 if word in document_words else 0 for word in vocabulary]


def build_bags_of_words(
    vocabulary_file: str = "vocabulario.txt",
    document_files: Iterable[str] = ("document.txt",)
) -> Dict[str, BagOfWords]:
    """
    given vocabulary and documents files names, return the bag of words of this
    vocabulary for each of the documents

    :param vocabulary_file:
    :param document_files: an iterable of document_words names
    :return: returns a dict of the type {file_name: bag_of_words of the file}
    """
    vocabulario = read_all_terms_from_file_to_lower(file_name=vocabulary_file)

    # in the end will be a dict with the key being the file name and the value will be
    # the bag of words of the file
    bags_of_words = {}
    for document in document_files:
        bags_of_words.update({
            document: bag_of_words(
                vocabulary=vocabulario,
                document_words=read_all_terms_from_file_to_lower(file_name=document)
            )
        })

    return bags_of_words


if __name__ == '__main__':
    result = build_bags_of_words()
    print(result)
