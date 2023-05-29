import random
import time
from utils import (
    calculate_vsm,
    build_vocabulary_from_files,
    get_all_files_in_directory,
    get_tf_idf,
)
from models import TfBuilder, IdfBuilder


# Generate a code that calculates the medium time of the execution of the VSM
# algorithm for random queries. The queries must be generated randomly, but
# they must be in the vocabulary. The code must be executed 100 times and the
# medium time must be calculated.
def questao5(documents_dir: str = "arquivos/musicas"):
    import nltk

    nltk.download("stopwords")

    stemmer = nltk.stem.PorterStemmer().stem
    stopwords = set(nltk.corpus.stopwords.words("english"))

    stopwords = {stemmer(stopword) for stopword in stopwords}

    # generate queries
    vocabulary = build_vocabulary_from_files(
        get_all_files_in_directory(directory_name=documents_dir),
        stopwords=stopwords,
    )

    # Generate 100 random queries based on the vocabulary
    # for _ in range(100):
    #     query = ""
    #     for _ in range(5):
    #         query += random.choice(list(vocabulary)) + " "
    #     queries.append(query)

    queries = [
        "slow dancing in a burning room",
        "like a stone on the water",
        "every once in a while i fall apart",
        "breath of the wild",
        "every day is exactly the same",
    ]

    times = []

    total_time = 0
    for query in queries:
        start_time = time.time()
        calculate_vsm(
            query=query,
            documents_dir=documents_dir,
            stopwords=stopwords,
            stemmer=stemmer,
        )
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        times.append(elapsed_time)

    print("===========================================")
    print("Total time: ", total_time)
    print("Medium time: ", total_time / len(queries))
    print("Median time: ", sorted(times)[len(times) // 2])

    print(times)


questao5()
