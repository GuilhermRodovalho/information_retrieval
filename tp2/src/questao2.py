"""
create a program that receive a vocabulary file and a path with documents
and it will do:
1 - calculate and exibit the tf_builder-idf_builder of each document 

"""
from utils import (
    get_all_files_in_directory,
    read_all_terms_from_file_to_lower,
    calculate_tf_idf,
)


def questao2(
    vocabulary_file: str = "vocabulario", documents_path: str = "arquivos/todo"
):
    # read the vocabulary
    vocabulary = read_all_terms_from_file_to_lower(file_name=vocabulary_file)

    files = get_all_files_in_directory(directory_name=documents_path)

    tf_idf = calculate_tf_idf(document_files=files, vocabulary=vocabulary)

    print(tf_idf)


questao2()
