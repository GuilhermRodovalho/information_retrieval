from utils import read_all_terms_from_file_to_lower, write_data_to_file


def questao1(input_file="input_q1.txt", output_file="output_q1.txt"):
    # leio todas as palavras do arquivo, ja transformadas para minusculas
    # uso o set para remover repetidos
    all_words = set(read_all_terms_from_file_to_lower(input_file))

    # retorno para lista, porque set nao tem conceito de posicao,
    # portanto nao eh possivel ordenar
    final_words = sorted(list(all_words))
    # escrevo as palavras ordenadas no arquivo
    write_data_to_file(file_name=output_file, words=final_words)
