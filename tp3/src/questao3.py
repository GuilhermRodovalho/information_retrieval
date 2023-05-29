import time
from typing import Dict, Tuple
from utils import calculate_vsm, print_sorted_similarities


def questao3(documents_path: str = "arquivos/todo"):
    # read the vocabulary
    import nltk

    nltk.download("stopwords")

    stopwords = set(nltk.corpus.stopwords.words("english"))

    similarities = calculate_vsm(
        query="to do",
        documents_dir=documents_path,
        vocab_file="vocabulario",
        stopwords=stopwords,
    )

    print("Similarities:")
    print_sorted_similarities(similarities)


questao3()
