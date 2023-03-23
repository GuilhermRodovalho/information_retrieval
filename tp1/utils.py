from typing import List, Iterable

from unidecode import unidecode


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
        for line in input_file.read().split("\n"):
            all_words += [
                remove_special_chars(term.lower())
                for term in line.strip().split(" ")
            ]

    return all_words


def write_data_to_file(
    file_name: str,
    words: Iterable[str]
) -> None:
    """
    Given a file_name, write the words to the file, one for each line

    :param file_name: a file that doesn't need to exist
    :param words: list of str to write to the file
    """
    with open(file=file_name, mode="w") as output:
        for word in words:
            output.write(word)
            output.write("\n")
