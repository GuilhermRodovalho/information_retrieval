import time
from utils import (
    get_all_files_in_directory,
    read_all_terms_from_file_to_lower,
    calculate_tf_idf,
    build_vocabulary_from_files,
)


def questao3(documents_path: str = "arquivos/musicas"):
    # read the vocabulary
    # vocabulary = read_all_terms_from_file_to_lower(file_name=vocabulary_file)

    files = get_all_files_in_directory(directory_name=documents_path)

    time_before = time.perf_counter()
    vocabulary = build_vocabulary_from_files(files)
    time_after = time.perf_counter()

    print("Took {} seconds to build the vocabulary".format(time_after - time_before))
    print("The vocabulary has {} terms".format(len(vocabulary)))

    time_before = time.perf_counter()
    tf_idf = calculate_tf_idf(document_files=files, vocabulary=vocabulary)
    time_after = time.perf_counter()
    print("Took {} seconds to calculate the tf-idf".format(time_after - time_before))

    largest_term, largest_tf_idf, document = "", 0, ""
    # find the largest tf-idf
    for file, terms in tf_idf.items():
        for term, val in terms.items():
            if val > largest_tf_idf:
                largest_tf_idf = val
                largest_term = term
                document = file

    print(
        "The largest tf-idf is {} for the term {} in the document {}".format(
            largest_tf_idf, largest_term, document
        )
    )

    print(tf_idf)

    # print(tf_idf)


questao3()
