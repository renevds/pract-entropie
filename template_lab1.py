import math
import numpy
import numpy as np
import matplotlib.pyplot as plt

SHREK_PATH = "text_shrek.txt"  # The full script of shrek
NEWS_PATH = "text_news.txt"  # A news article
PAPER_PATH = "text_paper.txt"  # A scientific paper


def get_entropy(data: list, memory: int = 0) -> float:
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


def get_distribution_output(data: list, name):
    chars, char_count = numpy.unique(data, return_counts=True)  # Get frequency's
    idx = np.argsort(char_count)
    chars = np.flip(chars[idx])[:50]
    char_count = np.flip(char_count[idx])[:50]

    fig, ax = plt.subplots()
    ax.bar(chars, char_count, width=1, edgecolor="white", linewidth=0.7)
    ax.xaxis.set_ticks(ticks=chars)
    ax.xaxis.set_ticklabels(ticklabels=chars, fontsize=8)
    plt.savefig(f'distributions_{name}.png', dpi=300, bbox_inches='tight')

def main():
    with open(SHREK_PATH, encoding="utf8") as f:  # Shrek
        shrek_data = list(f.read())
        get_distribution_output(shrek_data, "shrek")

    with open(NEWS_PATH, encoding="utf8") as f:  # News
        news_data = list(f.read())
        get_distribution_output(news_data, "news")

    with open(PAPER_PATH, encoding="utf8") as f:  # News
        paper_data = list(f.read())
        get_distribution_output(paper_data, "paper")


if __name__ == "__main__":
    main()
