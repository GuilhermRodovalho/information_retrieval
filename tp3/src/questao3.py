import time
from typing import Dict, Tuple
from utils import calculate_vsm


def questao3(documents_path: str = "arquivos/todo"):
    # read the vocabulary
    import nltk

    nltk.download("stopwords")

    stopwords = set(nltk.corpus.stopwords.words("english"))

    print(
        calculate_vsm(
            query="to do",
            documents_dir=documents_path,
            vocab_file="vocabulario",
            stopwords=stopwords,
        )
    )


questao3()
