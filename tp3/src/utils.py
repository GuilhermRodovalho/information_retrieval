import datetime
from math import sqrt
import os
import re
from typing import Callable, Dict, List, Iterable, Optional, Tuple, Collection, Union

from unidecode import unidecode

from models import IdfBuilder, TfBuilder


TF = Dict[str, float]
TF_IDF = Dict[str, float]


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


def read_all_terms_from_file_to_lower(
    file_name: str,
    remove_special_chars: Callable[[str], str] = remove_special_chars,
    stopwords: Collection[str] = [],
    stemmer: Callable[[str], str] = lambda x: x,
) -> List[str]:
    """
    Given a file name, reads all the content and return a list with
    all the words from the file in lowercase and parsed for special characters.
    Note the return may have duplicate words.

    :param file_name: name of the file
    :return: a list with the words
    """
    with open(file=file_name) as input_file:
        all_words = [
            stemmer(remove_special_chars(term.lower()))
            for line in input_file.readlines()
            for term in line.strip().split()
            if term not in stopwords
        ]

    return all_words


def write_data_to_file(file_name: str, words: Iterable[str]) -> None:
    """
    Given a file_name, write the words to the file, one for each line

    :param file_name: a file that doesn't need to exist
    :param words: list of str to write to the file
    """
    with open(file=file_name, mode="w") as output:
        output.write("\n".join(words))


def build_vocabulary_from_files(
    files: Iterable[str] = ("document.txt",),
    stemmer: Callable[[str], str] = lambda x: x,
    stopwords: Collection[str] = [],
) -> List[str]:
    """
    Given a list of files, returns a vocabulary with all the words from the files

    :param files: a list of files names
    :return: a list of words
    """
    vocabulary = set()

    for file in files:
        vocabulary.update(
            set(
                read_all_terms_from_file_to_lower(
                    file_name=file, stemmer=stemmer, stopwords=stopwords
                )
            )
        )

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


def build_tf_idf_by_file_dict(
    tf_by_file: Dict[str, TF], idf_builder: IdfBuilder
) -> Dict[str, Dict[str, float]]:
    tf_idf_by_file: Dict[str, Dict[str, float]] = {}

    # calculate the tf_idf by file
    for file, tf in tf_by_file.items():
        tf_idf_by_file.update(
            {file: {word: tf[word] * idf_builder[word] for word in tf.keys()}}
        )

    return tf_idf_by_file


def calculate_tf_idf(
    vocabulary: List[str],
    query: Optional[str] = None,
    document_files: Iterable[str] = ("document.txt",),
    stopwords: Collection[str] = [],
    stemmer: Callable[[str], str] = lambda x: x,
) -> Tuple[Dict[str, TF_IDF], IdfBuilder]:
    """
    Given a dict of bags of words and the files names, calculate the tf-idf of each
    """
    idf_builder = IdfBuilder(vocabulary=vocabulary, stopwords=stopwords)
    tf_by_file: Dict[str, TF] = {}

    tf_builder = TfBuilder(stopwords=stopwords)
    query_tf = {}
    if query:
        query_terms = [
            stemmer(remove_special_chars(term.lower()))
            for term in query.strip().split()
        ]
        query_tf = tf_builder.calculate_tf(query_terms, vocabulary)

    for file in document_files:
        # considering that we are iterating through the file content twice
        # we could optimize this by creating a function that returns both
        # but for now it's ok
        file_content = read_all_terms_from_file_to_lower(
            file_name=file,
            stopwords=stopwords,
            stemmer=stemmer,
        )
        tf = tf_builder.calculate_tf(file_content, vocabulary)
        idf_builder.add_document_terms(set(file_content))

        tf_by_file.update({file: tf})

    idf_builder.calculate_idf()

    # calculate the tf_idf by file
    tf_idf_by_file: Dict[str, Dict[str, float]] = build_tf_idf_by_file_dict(
        tf_by_file=tf_by_file, idf_builder=idf_builder
    )

    if query:
        # Calculate tf-idf for query terms
        query_tf_idf = build_tf_idf_by_file_dict(
            tf_by_file={"query": query_tf}, idf_builder=idf_builder
        )

        tf_idf_by_file.update(query_tf_idf)

    # Returning also the idf_builder so it's easier to calculate the tf-idf of a new
    # query
    return tf_idf_by_file, idf_builder


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


def calculate_cross_product_and_norm(
    first: TF_IDF, second: TF_IDF
) -> Tuple[float, float, float]:
    cross_product = 0

    norm_first = 0
    norm_second = 0

    for word, tf_idf in first.items():
        cross_product += tf_idf * second[word]
        norm_first += tf_idf**2
        norm_second += second[word] ** 2

    return cross_product, sqrt(norm_first), sqrt(norm_second)


def calculate_vsm(
    query: str,
    documents_dir: str = "./arquivos/todo",
    vocab_file: str = "vocabulario",
    vocabulary: List[str] = [],
    stopwords: Collection[str] = [],
    stemmer: Callable[[str], str] = lambda x: x,
    tf_idf: Dict[str, Dict[str, float]] = {},
    idf_builder: Optional[IdfBuilder] = None,
    tf_builder: Optional[TfBuilder] = None,
) -> Dict[str, float]:
    """
    Given a vocabulary file, a directory with documents and a query, calculate the
    vsm (vectorial space model) of the query with the documents

    :param vocab_file: the name of the file with the vocabulary
    :param documents_dir: the name of the directory with the documents
    :param query: the user query
    :return: a dict of the type {document_name: vsm}
    """
    # read the vocabulary

    if tf_idf and not idf_builder:
        raise Exception("tf_idf and idf_builder must be passed together")

    if not vocabulary:
        vocabulary = read_all_terms_from_file_to_lower(
            file_name=vocab_file,
            stopwords=stopwords,
            stemmer=stemmer,
        )

    if not tf_idf or not idf_builder:
        tf_idf, idf_builder = calculate_tf_idf(
            document_files=get_all_files_in_directory(documents_dir),
            vocabulary=vocabulary,
            stopwords=stopwords,
            stemmer=stemmer,
        )

    if not tf_builder:
        tf_builder = TfBuilder(stopwords=stopwords, stemmer=stemmer)

    if "query" in tf_idf:
        query_tf_idf = tf_idf["query"]
        del tf_idf["query"]
    else:
        # calculate tf-idf for the query
        query_terms = query.split()
        query_tf = tf_builder.calculate_tf(
            words=map(stemmer, query_terms), vocabulary=vocabulary
        )
        query_tf_idf = {
            word: query_tf[word] * idf_builder[word] for word in query_tf.keys()
        }

    similarities = {}

    for doc, this_tf_idf in tf_idf.items():
        cross_product, doc_norm, query_norm = calculate_cross_product_and_norm(
            first=this_tf_idf, second=query_tf_idf
        )

        if doc_norm == 0 or query_norm == 0:
            similarities[doc] = 0
        else:
            similarities[doc] = cross_product / (doc_norm * query_norm)

    return similarities
