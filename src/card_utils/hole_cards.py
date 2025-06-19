import numpy as np
from numba import jit

from src.card_utils.card import get_rank, get_suit

OFF_SUIT = np.uint8(0)
SUITED = np.uint8(1)
PAIR = np.uint8(2)

@jit(nopython=True)
def get_2d_index(card1: np.uint8, card2: np.uint8) -> np.uint8:
    row, col = get_index(card1, card2)
    return np.uint8(row * 13 + col)

@jit(nopython=True)
def get_index(card1: np.uint8, card2: np.uint8) -> tuple[np.uint8, np.uint8]:
    low_rank = get_rank(card1)
    high_rank = get_rank(card2)
    if low_rank > high_rank:
        tmp = low_rank
        low_rank = high_rank
        high_rank = tmp

    suits_match = get_suit(card1) == get_suit(card2)

    row = high_rank if suits_match else low_rank
    col = low_rank if suits_match else high_rank

    return row, col

@jit(nopython=True)
def get_hole_code_from_2d(row: np.uint8, col: np.uint8) -> tuple[np.uint8, np.uint8, np.uint8]:
    # pair
    if row == col:
        return row, col, np.uint8(PAIR)
    # suits match
    elif row > col:
        return row, col, np.uint8(SUITED)
    # off suit
    else:
        return row, col, np.uint8(OFF_SUIT)
