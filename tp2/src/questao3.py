import time
from typing import Dict, Tuple
from utils import (
    get_all_files_in_directory,
    read_all_terms_from_file_to_lower,
    calculate_tf_idf,
    build_vocabulary_from_files,
)


def largest_tf_idf(tf_idf: Dict[str, Dict[str, float]]) -> Tuple[str, str, float]:
    largest_term, largest_tf_idf, document = "", 0, ""
    second_largest_term, second_largest_tf_idf, second_document = "", 0, ""
    # find the largest tf-idf
    for file, terms in tf_idf.items():
        for term, val in terms.items():
            if val > largest_tf_idf:
                second_largest_tf_idf = largest_tf_idf
                second_largest_term = largest_term
                second_document = document

                largest_tf_idf = val
                largest_term = term
                document = file
            elif val > second_largest_tf_idf:
                second_largest_tf_idf = val
                second_largest_term = term
                second_document = file

    return largest_term, document, largest_tf_idf


def questao3(documents_path: str = "arquivos/musicas"):
    # read the vocabulary
    # vocabulary = read_all_terms_from_file_to_lower(file_name=vocabulary_file)

    files = get_all_files_in_directory(directory_name=documents_path)

    time_before = time.perf_counter()
    vocabulary = build_vocabulary_from_files(files)
    time_after = time.perf_counter()

    vocabulary_time = time_after - time_before

    time_before = time.perf_counter()
    tf_idf = calculate_tf_idf(document_files=files, vocabulary=vocabulary)
    time_after = time.perf_counter()

    tf_idf_time = time_after - time_before

    print(tf_idf)

    # print the eileen tf-idf
    print("Eileen tf-idf")
    print(tf_idf["arquivos/musicas/come_on_eileen.txt"]["eileen"])

    largest_term, largest_value, document = largest_tf_idf(tf_idf)
    print("Took {} seconds to build the vocabulary".format(vocabulary_time))
    print("The vocabulary has {} terms".format(len(vocabulary)))
    print("Took {} seconds to calculate the tf-idf".format(tf_idf_time))
    print(
        "The largest tf-idf is {} for the term {} in the document {}".format(
            largest_value, largest_term, document
        )
    )


questao3()
