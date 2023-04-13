from utils import (
    build_vocabulary_from_files,
    get_all_files_in_directory,
    write_data_to_file,
    build_bags_of_words,
)


# TODO: fazer com que pegue todos os arquivos dos subdiretórios também
def questao1(input_dir="arquivos", output_file="output_q1.txt"):
    files = get_all_files_in_directory(input_dir)

    vocabulary = build_vocabulary_from_files(files)
    write_data_to_file(output_file, vocabulary)

    bags = build_bags_of_words(vocabulary, files)

    print(bags)


questao1()
