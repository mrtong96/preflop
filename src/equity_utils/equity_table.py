import itertools
import time
from collections import Counter

import numpy as np
import pandas as pd
from sympy.stats.sampling.sample_numpy import numpy

from src.card_utils.card import get_rank, get_suit
from src.equity_utils.head_to_head_equity import get_head_to_head_equity


# write to tmp... don't be stupid
INPUT_PATH = '../resources/equity_raw_data.txt'
OUTPUT_PATH = '../resources/equity_raw_data.tmp.txt'
OUTPUT_NUMPY_PATH = '../resources/equity.npy'

# who cares if it's slow
def get_suit_rotation(cards: tuple[int, int], other_cards_array: tuple[int, int]) -> tuple[str, ...]:
    combined_combo = list(sorted(cards)) + list(sorted(other_cards_array))
    combined_combo = np.array(combined_combo, dtype=np.int8)
    ranks = [get_rank(card) for card in combined_combo]
    suits = []
    suit_mapping = dict()
    for card in combined_combo:
        if get_suit(card) not in suit_mapping:
            suit_mapping[get_suit(card)] = len(suit_mapping)
        suits.append(suit_mapping[get_suit(card)])

    cards = [rank << 2 | suit for rank, suit in zip(ranks, suits)]

    return tuple(cards)

def raw_text_to_numpy_file():
    """
    Do it in two stages because it's expensive to read raw text
    :return:
    """

    header_cols = None
    data = []
    for line in open(INPUT_PATH, 'r').readlines():
        if header_cols is None:
            header_cols = [col.strip() for col in line.split(',')]
        elif line.strip():
            data.append(tuple(eval(line)))
        else:
            break

    equity_df = pd.DataFrame(data, columns=header_cols)

    equity_array = np.zeros((len(equity_df), 4 + 1 + 3), dtype=np.int32)
    for i, (suit_rotation, num_combos, win_rates) in enumerate(equity_df.values):
        equity_array[i, 0: 4] = suit_rotation
        equity_array[i, 4] = num_combos
        equity_array[i, 5: 8] = win_rates

    np.save(f'../resources/{OUTPUT_NUMPY_PATH}', equity_array)

def load_head_to_head_equity():
    """
    Return the data as a pandas dataframe
    :return:
    """
    numpy_data = np.load(f'../resources/{OUTPUT_NUMPY_PATH}')

    card_tuples = numpy_data[:, 0: 4]
    num_combos = numpy_data[:, 4]
    win_rates = numpy_data[:, 5: 8]

    return pd.DataFrame(list(zip(card_tuples, num_combos, win_rates)), columns=['card_tuple', 'combos', 'win_rates'])

def main():
    """
    Compute the head-to-head equity of all ncr(52, 2) * ncr(50, 2) cards

    Uses suit rotations to reduce the problem space to ~100k calculations.

    Takes ~6 or so hours to compute everything
    :return:
    """
    suit_rotation_count = Counter()

    deck = list(range(52))
    for cards in itertools.combinations(deck, 2):
        remaining_cards = [card for card in deck if card not in cards]

        for other_cards in itertools.combinations(remaining_cards, 2):
            suit_rotation = get_suit_rotation(cards, other_cards)
            suit_rotation_count[suit_rotation] += 1

    t0 = time.time()
    with open(OUTPUT_PATH, 'w') as file:
        for iter, (key, count) in enumerate(suit_rotation_count.items()):
            card_vector = np.array(key, dtype=np.int8)
            equity = get_head_to_head_equity(card_vector[:2], card_vector[2:])

            text = f'{key}, {count}, {list(equity)}'
            file.write(text + '\n')

            if iter % 100 == 0:
                print(f'iter: {iter}, time: {time.time() - t0}')
                file.flush()
        file.flush()

    raw_text_to_numpy_file()

if __name__ == '__main__':
    main()