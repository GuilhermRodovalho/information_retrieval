from utils import calculate_vsm, print_sorted_similarities


def questao2(
    vocab_file: str = "vocabulario",
    documents_path: str = "arquivos/todo",
    query: str = "to do",
):
    # read the vocabulary

    similarities = calculate_vsm(
        query=query,
        documents_dir=documents_path,
        vocab_file=vocab_file,
    )

    print("Similarities:")
    print_sorted_similarities(similarities)


questao2()
