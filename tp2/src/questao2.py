"""
create a program that receive a vocabulary file and a path with documents
and it will do:
1 - calculate and exibit the tf-idf of each document 

"""

import os

from utils import (
    build_bags_of_words,
    read_all_terms_from_file_to_lower,
    calculate_tf_idf,
    write_data_to_file,
)


def questao2(vocabulary_file: str, documents_path: str, output_file: str):
    # read the vocabulary
    vocabulary = read_all_terms_from_file_to_lower(file_name=vocabulary_file)
    # read all files from the directory
    root, _, files = next(os.walk(documents_path))

    # add the root folder before the filename
    files = list(map(lambda file: root + "/" + file, files))

    bags = build_bags_of_words(vocabulary, files)

    # calculate the tf-idf of each document
    tf_idf = calculate_tf_idf(bags, files)

    # write the tf-idf to a file
    write_data_to_file(output_file, tf_idf)

    print(tf_idf)
