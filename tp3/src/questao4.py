from typing import Dict
from utils import (
    calculate_vsm,
    build_vocabulary_from_files,
    get_all_files_in_directory,
    calculate_tf_idf,
    build_tf_idf_by_file_dict,
)
from models import TfBuilder


def questao4(documents_path: str = "arquivos/musicas"):
    # read the vocabulary
    import nltk

    nltk.download("stopwords")

    stemmer = nltk.stem.PorterStemmer().stem
    stopwords = set(nltk.corpus.stopwords.words("english"))

    vocabulary = build_vocabulary_from_files(
        get_all_files_in_directory(directory_name=documents_path),
        stopwords=stopwords,
    )

    vocabulary_with_stem = build_vocabulary_from_files(
        get_all_files_in_directory(directory_name=documents_path),
        stemmer=stemmer,
        stopwords=stopwords,
    )

    # calculate tf-idf for all documents before hand so we don't have to do it for each query
    tf_idf_without_stemming, idf_builder_without_stemming = calculate_tf_idf(
        document_files=get_all_files_in_directory(directory_name=documents_path),
        vocabulary=vocabulary,
        stopwords=stopwords,
    )
    tf_idf_with_stemming, idf_builder_with_stemming = calculate_tf_idf(
        document_files=get_all_files_in_directory(directory_name=documents_path),
        vocabulary=vocabulary_with_stem,
        stopwords=stopwords,
        stemmer=stemmer,
    )

    queries = [
        "slow dancing in a burning room",
        "like a stone on the water",
        "every once in a while i fall apart",
        "breath of the wild",
        "every day is exactly the same",
    ]

    tf_builder_without_stemming = TfBuilder(stopwords=stopwords)
    tf_builder_with_stemming = TfBuilder(stopwords=stopwords, stemmer=stemmer)

    # check if the words are in the vocabulary

    for query in queries[0:1]:
        print("===========================================")
        print("Query: ", query)

        # calculate the vsm for the query and all documents
        similarities = calculate_vsm(
            query=query,
            documents_dir=documents_path,
            vocabulary=vocabulary,
            stopwords=stopwords,
            tf_idf=tf_idf_without_stemming,
            tf_builder=tf_builder_without_stemming,
        )

        print("Similaridades sem stemming")
        print_top5_similarities(similarities)

        similarities_with_stemming = calculate_vsm(
            query=query,
            documents_dir=documents_path,
            vocabulary=vocabulary_with_stem,
            stopwords=stopwords,
            stemmer=stemmer,
            tf_idf=tf_idf_with_stemming,
            tf_builder=tf_builder_with_stemming,
        )

        print("Similaridades com stemming")
        print_top5_similarities(similarities=similarities_with_stemming)


def print_top5_similarities(similarities: Dict[str, float]) -> None:
    import operator

    top5 = dict(
        sorted(similarities.items(), key=operator.itemgetter(1), reverse=True)[:5]
    )

    for item, val in top5.items():
        print(f"{item}: {val}")


questao4()
