import time
from utils import (
    calculate_vsm,
    calculate_tf_idf,
    build_vocabulary_from_files,
    get_all_files_in_directory,
)


def questao2(documents_path: str = "arquivos/todo"):
    # read the vocabulary

    print(
        calculate_vsm(
            query="to do",
            documents_dir=documents_path,
            vocab_file="vocabulario",
        )
    )


questao2()
