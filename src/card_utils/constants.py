import itertools

import numpy as np

from src.card_utils.card import RANKS

STRAIGHT_FLUSH = 7  # bottom 4 bits for the highest card
FOUR_KIND = 7  # 8 bits for the 4/1 cards
FULL_HOUSE = 6  # 8 bits for the 3/2 cards
FLUSH = 5  # 13 bits for the bit map of cards
STRAIGHT = 4  # 4 bits for the high card
THREE_KIND = 3  # 12 bits for the 3/1/1 cards
TWO_PAIR = 2  # 12 bits for the 2/2/1 cards
ONE_PAIR = 1   # 2^13
HIGH_CARD = 0  # 13 bits for the bit map
three_card_combos = list(itertools.combinations(range(len(RANKS)), 3))
THREE_CARD_TO_CODE = np.zeros((13, 13, 13), dtype=np.uint16)

three_card_combos = [sorted(combo, reverse=True) for combo in three_card_combos]
three_card_combos.sort()
for i, combo in enumerate(three_card_combos):
    for combo_permutation in itertools.permutations(combo):
        THREE_CARD_TO_CODE[*combo] = i + 1


if __name__ == '__main__':
    print(len(three_card_combos))

    for el in three_card_combos:
        print(el)