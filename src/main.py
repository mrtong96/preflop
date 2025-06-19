import time
from collections import Counter

import numpy as np

import pandas as pd

from src.card_utils.card import get_card_str, get_card
from src.card_utils.combinatorics_utils import ncr
from src.card_utils.hole_cards import get_2d_index
from src.card_utils.rank_spec import RankSpec
from src.equity_utils.joint_equity_probabilities import load_head_to_head_joint_probability, \
    load_head_to_head_wins_probability, compute_equity, compute_weighted_equity, compute_weighted_equity_vector


def debug():
    shared_cards = ['TC', '9C', '8S', 'QS']
    p1_cards = ['5C', '6C'] + shared_cards
    p2_cards = ['KD', 'KH'] + shared_cards
    p1_cards = np.array([get_card(card) for card in p1_cards], dtype=np.int8)
    p2_cards = np.array([get_card(card) for card in p2_cards], dtype=np.int8)

    missing_cards = [i for i in range(52) if i not in p1_cards and i not in p2_cards]
    starting_community_cards = np.array(missing_cards[:7 - len(p1_cards)], dtype=np.int8)

    p1_rank_spec = RankSpec(p1_cards, p2_cards, starting_community_cards)
    p2_rank_spec = RankSpec(p2_cards, p1_cards, starting_community_cards)

    p1_wins_counter = 0
    p2_wins_counter = 0
    tie_counter = 0

    while True:
        p1_rank = p1_rank_spec.get_hand_rank()
        p2_rank = p2_rank_spec.get_hand_rank()

        if p1_rank > p2_rank:
            p1_wins_counter += 1
        elif p2_rank > p1_rank:
            p2_wins_counter += 1
        else:
            tie_counter += 1

            print(hex(p1_rank), [get_card_str(card) for card in p1_rank_spec.card_vector])
            print(hex(p2_rank), [get_card_str(card) for card in p2_rank_spec.card_vector])
            print('-' * 80)

        p1_iter = p1_rank_spec.attempt_rotation()
        p2_iter = p2_rank_spec.attempt_rotation()

        if p1_iter != p2_iter:
            raise RuntimeError("Iteration bug")
        elif not p1_iter:
            break

    print(p1_wins_counter, p2_wins_counter, tie_counter)


def main():
    data = load_head_to_head_joint_probability()
    print(np.max(data), np.min(data))
    print(np.sum(data == 6))


if __name__ == '__main__':
    main()