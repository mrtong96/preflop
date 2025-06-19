from numba import jit

import numpy as np

from src.card_utils.constants import STRAIGHT_FLUSH
from src.card_utils.rank_spec import RankSpec


@jit(nopython=True)
def get_head_to_head_equity(p1_cards: np.array, p2_cards: np.array) -> np.array:
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

        p1_iter = p1_rank_spec.attempt_rotation()
        p2_iter = p2_rank_spec.attempt_rotation()

        if p1_iter != p2_iter:
            raise RuntimeError("Iteration bug")
        elif not p1_iter:
            break

    return np.array([p1_wins_counter, p2_wins_counter, tie_counter], dtype=np.int32)

def get_rank_dist(cards: np.array, block_cards):
    missing_cards = [i for i in range(52) if i not in cards and i not in block_cards]
    starting_community_cards = np.array(missing_cards[:5], dtype=np.int8)

    rank_spec = RankSpec(cards, block_cards, starting_community_cards)
    rank_counts = np.zeros(STRAIGHT_FLUSH + 2, dtype=np.int32)

    while True:
        rank = rank_spec.get_hand_rank()

        rank_type = rank >> 13
        if rank_type == STRAIGHT_FLUSH and (rank >> 12) & 1:
            rank_type += 1
        rank_counts[rank_type] += 1

        if not rank_spec.attempt_rotation():
            break

    return rank_counts[::-1]
