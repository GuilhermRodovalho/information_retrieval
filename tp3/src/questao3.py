import time
from typing import Dict, Tuple
from utils import calculate_vsm_using_nltk, calculate_vsm


def questao2(documents_path: str = "arquivos/todo"):
    # read the vocabulary
    print(
        calculate_vsm_using_nltk(
            query="to do", documents_dir=documents_path, vocab_file="vocabulario"
        )
    )


questao2()
