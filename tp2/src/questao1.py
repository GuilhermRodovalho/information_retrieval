import os

from utils import build_vocabulary_from_files, write_data_to_file, build_bags_of_words


def questao1(input_dir="arquivos", output_file="output_q1.txt"):
    # read all files from the directory
    root, _, files = next(os.walk(input_dir))

    # add the root folder before the filename
    files = list(map(lambda file: root + "/" + file, files))

    vocabulary = build_vocabulary_from_files(files)
    write_data_to_file(output_file, vocabulary)

    bags = build_bags_of_words(vocabulary, files)

    print(bags)


questao1()
