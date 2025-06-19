# who cares if it's slow
import itertools
from collections import Counter

import numpy as np

from src.card_utils.card import get_rank, get_suit


def get_suit_rotation(cards: tuple[int, int, int]) -> tuple[str, ...]:
    cards = np.array(cards, dtype=np.int8)
    ranks = [get_rank(card) for card in cards]
    suits = []
    suit_mapping = dict()
    for card in cards:
        if get_suit(card) not in suit_mapping:
            suit_mapping[get_suit(card)] = len(suit_mapping)
        suits.append(suit_mapping[get_suit(card)])

    cards = [rank << 2 | suit for rank, suit in zip(ranks, suits)]
    return tuple(cards)

def main():
    suit_rotation_count = Counter()

    deck = list(range(52))
    for cards in itertools.combinations(deck, 3):
        suit_rotation = get_suit_rotation(cards)
        suit_rotation_count[suit_rotation] += 1

    print(len(suit_rotation_count), sum(suit_rotation_count.values()))

if __name__ == '__main__':
    main()