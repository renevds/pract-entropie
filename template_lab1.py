import math
import numpy
import numpy as np
import matplotlib.pyplot as plt

SHREK_PATH = "text_shrek.txt"  # The full script of shrek
NEWS_PATH = "text_nieuws.txt"  # A news article
PAPER_PATH = "text_paper.txt"  # A scientific paper

SHREK_PATH_ZIPPED = "text_shrek.7z"  # The full script of shrek zipped
NEWS_PATH_ZIPPED = "text_nieuws.7z"  # A news article zipped
PAPER_PATH_ZIPPED = "text_paper.7z"  # A scientific paper zipped

ENTROPY_RANGE = 25  # Max memory to calculate entropy for
MEMORY_STEP_SIZE = 1  # Step size for memory in entropy calculation

SHREK_ENTROPY_TITLE = f'Figuur 2.1, Verloop van de entropie in functie van het geheugen voor \'shrek\' '
NEWS_ENTROPY_TITLE = f'Figuur 2.2, Verloop van de entropie in functie van het geheugen voor \'nieuws\' '
PAPER_ENTROPY_TITLE = f'Figuur 2.3, Verloop van de entropie in functie van het geheugen voor \'paper\' '

SHREK_DIST_TITLE = f'Figuur 1.1, Distributie van de karakters voor \'shrek\''
NEWS_DIST_TITLE = f'Figuur 1.2, Distributie van de karakters voor \'nieuws\''
PAPER_DIST_TITLE = f'Figuur 1.3, Distributie van de karakters voor \'paper\''


def get_entropy(data: list, memory: int = 0) -> float:
    assert memory >= 0, "Memory should be positive"
    if not isinstance(data[0], str):  # In case we receive binary
        assert memory == 0, "Calculating entropy of bytes with memory not supported"
        assert isinstance(data[0], bytes), "Type in list not supported"
        _, byte_count = numpy.unique(data, return_counts=True)  # Get frequency's
        byte_chance = byte_count / len(data)
        return -numpy.sum(byte_chance*np.log2(byte_chance))

    arr = numpy.array(data)  # Using a numpy array for performance

    as_strided = numpy.lib.stride_tricks.as_strided
    all_full_chunks = as_strided(data, (arr.size - memory, memory + 1), arr.strides * 2)  # Chunks of size memory + 1
    all_full_chunks = np.array([''.join(i) for i in all_full_chunks])

    all_sub_chunks = as_strided(data, (arr.size - memory, memory), arr.strides * 2)  # Chunks of size memory
    all_sub_chunks = np.array([''.join(i) for i in all_sub_chunks])

    full_chunks, full_chunk_count = numpy.unique(all_full_chunks, return_counts=True)  # Frequency of full chunks
    full_chunk_chance = full_chunk_count / all_full_chunks.size  # Chance of full chunks
    sub_chunks, sub_chunk_count = numpy.unique(all_sub_chunks, return_counts=True)  # Frequency of sub chunks

    sub_chunk_map = dict(zip(sub_chunks, sub_chunk_count))  # Place the sub chunks in a hashmap

    results = 0
    for (index, chunk) in enumerate(full_chunks):
        results += full_chunk_chance[index] * math.log2(
            full_chunk_count[index] / sub_chunk_map[chunk[:-1]])  # Calculate entropie
    return -results


def get_distribution_output(data: list, name: str, title: str, color: str):
    chars, char_count = numpy.unique(data, return_counts=True)  # Get frequency's
    idx = np.argsort(char_count)
    chars = np.flip(chars[idx])[:50]
    char_count = np.flip(char_count[idx])[:50]  # Sort characters by total count and only keep first 50

    # Plot this result
    fig, ax = plt.subplots()
    ax.bar(chars, char_count, width=1, edgecolor="white", linewidth=0.7, color=color)
    ax.xaxis.set_ticks(ticks=chars)
    plt.xlabel("Karakter")
    plt.ylabel("Aantal")
    plt.title(title)
    ax.xaxis.set_ticklabels(ticklabels=chars, fontsize=8)
    plt.savefig(f'distributions_{name}.png', dpi=300, bbox_inches='tight')


def get_entropy_output(data: list, name: str, title: str, color: str):
    results = []
    entropy_range = range(0, ENTROPY_RANGE + 1, MEMORY_STEP_SIZE)
    for i in entropy_range:
        results.append(get_entropy(data, i))

    # Plot this result
    fig, ax = plt.subplots()
    ax.plot(entropy_range, results, color=color)
    plt.xlabel("Geheugen")
    plt.ylabel("Entropie")
    plt.title(title)
    plt.savefig(f'entropy_{name}.png', dpi=300, bbox_inches='tight')


def read_bytes(f):
    byte = f.read(1)
    res = []
    while byte:
        res.append(byte)
        byte = f.read(1)
    return res


def generate_shrek():
    print("SHREK")
    with open(SHREK_PATH, encoding="utf8") as f:  # Shrek
        shrek_data = list(f.read())
        get_distribution_output(shrek_data, "shrek", SHREK_DIST_TITLE, 'blue')  # Get character distribution
        get_entropy_output(shrek_data, "shrek", SHREK_ENTROPY_TITLE, 'blue')  # Get entropy for different memory sizes

    with open(SHREK_PATH, "rb") as f:  # Get entropy for byte-read file
        shrek_data = read_bytes(f)
        print(get_entropy(shrek_data, 0))

    with open(SHREK_PATH_ZIPPED, "rb") as f:  # Get entropy for byte-read zipped file
        shrek_data = read_bytes(f)
        print(get_entropy(shrek_data, 0))


def generate_news():
    print("NEWS")
    with open(NEWS_PATH, encoding="utf8") as f:  # News
        news_data = list(f.read())
        get_distribution_output(news_data, "news", NEWS_DIST_TITLE,
                                'green')
        get_entropy_output(news_data, "news", NEWS_ENTROPY_TITLE, 'green')

    with open(NEWS_PATH, "rb") as f:  # Get entropy for byte-read file
        news_data = read_bytes(f)
        print(get_entropy(news_data, 0))

    with open(NEWS_PATH_ZIPPED, "rb") as f:  # Get entropy for byte-read zipped file
        news_data = read_bytes(f)
        print(get_entropy(news_data, 0))


def generate_paper():
    print("PAPER")
    with open(PAPER_PATH, encoding="utf8") as f:  # Paper
        paper_data = list(f.read())
        get_distribution_output(paper_data, "paper", PAPER_ENTROPY_TITLE,
                                'orange')
        get_entropy_output(paper_data, "paper", PAPER_ENTROPY_TITLE, 'orange')

    with open(PAPER_PATH, "rb") as f:  # Get entropy for byte-read file
        paper_data = read_bytes(f)
        print(get_entropy(paper_data, 0))

    with open(PAPER_PATH_ZIPPED, "rb") as f:  # Get entropy for byte-read zipped file
        paper_data = read_bytes(f)
        print(get_entropy(paper_data, 0))


def main():
    generate_news()
    generate_paper()
    generate_shrek()


if __name__ == "__main__":
    main()
