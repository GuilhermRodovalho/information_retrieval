import os
from typing import Dict, List, Iterable

from unidecode import unidecode

from models import IdfBuilder, TfBuilder


# TODO: criar um arquivo para parsing de caracteres
def remove_special_chars(word: str) -> str:
    """
    Return the string without any special characters, keeping only alphabetial.

    :param word: palavra a ser parseadas
    :return: palavra sem caracteres especiais.
    """
    string = ""
    for c in word:
        if c.isalpha():
            c = unidecode(c)
            string += c

    return string


# TODO: criar classe auxiliar para input/output de arquivos?
def read_all_terms_from_file_to_lower(file_name: str) -> List[str]:
    """
    Given a file name, reads all the content and return a list with
    all the words from the file in lowercase and parsed for special characters.
    Note the return may have duplicate words.

    :param file_name: name of the file
    :return: a list with the words
    """
    with open(file=file_name) as input_file:
        all_words = []
        for line in input_file.read().split("\n"):  # breaking into lines
            all_words += [
                remove_special_chars(term.lower())
                for term in line.strip().split()  # strip used for removing trailing white spaces and spliting into words
                if term
            ]

    return all_words


def write_data_to_file(file_name: str, words: Iterable[str]) -> None:
    """
    Given a file_name, write the words to the file, one for each line

    :param file_name: a file that doesn't need to exist
    :param words: list of str to write to the file
    """
    with open(file=file_name, mode="w") as output:
        for word in words:
            output.write(word)
            output.write("\n")


def build_vocabulary_from_files(
    files: Iterable[str] = ("document.txt",),
) -> List[str]:
    """
    Given a list of files, returns a vocabulary with all the words from the files

    :param files: a list of files names
    :return: a list of words
    """
    vocabulary = set()
    for file in files:
        vocabulary.update(set(read_all_terms_from_file_to_lower(file_name=file)))

    return list(vocabulary)


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
    vocabulary: List[str],
    document_files: Iterable[str] = ("document.txt",),
) -> Dict[str, BagOfWords]:
    """
    given vocabulary and documents files names, return the bag of words of this
    vocabulary for each of the documents

    :param vocabulary_file:
    :param document_files: an iterable of document_words names
    :return: returns a dict of the type {file_name: bag_of_words of the file}
    """
    # in the end will be a dict with the key being the file name and the value will be
    # the bag of words of the file
    bags_of_words = {}
    for document in document_files:
        document_words = read_all_terms_from_file_to_lower(file_name=document)
        bags_of_words.update(
            {
                document: bag_of_words(
                    vocabulary=vocabulary,
                    document_words=document_words,
                )
            }
        )

    return bags_of_words


def calculate_tf_idf(
    vocabulary: List[str],
    document_files: Iterable[str] = ("document.txt",),
) -> Dict[str, Dict[str, float]]:
    """
    Given a dict of bags of words and the files names, calculate the tf-idf of each
    """
    idf_builder = IdfBuilder(vocabulary=vocabulary)
    tf_by_file = {}

    for file in document_files:
        file_content = read_all_terms_from_file_to_lower(file_name=file)
        # TODO
        # considering that we are iterating through the file content twice
        # we could optimize this by creating a function that returns both
        # but for now it's ok
        tf = TfBuilder.calculate_tf(file_content)
        idf_builder.add_document_terms(set(file_content))

        tf_by_file.update({file: tf})

    idf_builder.calculate_idf()

    tf_idf_by_file: Dict[str, Dict[str, float]] = {}

    # calculate the tf_idf by file
    for file, tf in tf_by_file.items():
        tf_idf_by_file.update(
            {
                file: {
                    word: tf[word] * idf_builder[word]
                    for word in tf.keys()
                    if idf_builder[word] != 0
                }
            }
        )

    return tf_idf_by_file


def get_all_files_in_directory(directory_name: str) -> List[str]:
    """
    Given a directory name, return all the files in the directory with the full path

    :param directory_name: the name of the directory
    :return: a list of files with the full path
    """
    # read all files from the directory
    files_names = []

    for root, _, files in os.walk(directory_name):
        for file in files:
            files_names.append(os.path.join(root, file))

    return files_names
