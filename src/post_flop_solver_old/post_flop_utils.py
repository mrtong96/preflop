import functools

import numpy as np
from numba import jit

from src.card_utils.combinatorics_utils import ncr

VECTOR_SIZE = int(ncr(52, 2))
# just cache the results instead of being smart with pyramid sums
ROW_OFFSETS = np.cumsum([0] + [51 - i for i in range(51)])

@jit(nopython=True)
def get_vector_index(card1: np.int8, card2: np.int8) -> int:
    """
    Function to map a combination of 2 cards to a ncr(52, 2) index.
    """
    lower_card = int(min(card1, card2))
    upper_card = int(max(card1, card2))

    # pyramid sum. First row has 51 elements, second has 50, etc.
    return ROW_OFFSETS[lower_card] + upper_card - lower_card - 1

@jit(nopython=True)
def get_vector_indexes_with_card(card: np.int8) -> np.array:
    """
    Given a card, get all the indexes that have that card in it
    """
    return np.array([get_vector_index(card1=card, card2=np.int8(other))
                     for other in range(52) if card != other])

# cache the results in a matrix
VECTOR_INDICES_MATRIX = np.array([get_vector_indexes_with_card(np.int8(card)) for card in range(52)])

@jit(nopython=True)
def get_cards_from_vector_index(index: int) -> np.array:
    """
    convert from the vector index to the actual two cards we have
    """
    lower_card = np.searchsorted(ROW_OFFSETS, index, side='right') - 1
    upper_card = index - ROW_OFFSETS[lower_card] + lower_card + 1
    return np.array([lower_card, upper_card], dtype=np.int8)

@jit(nopython=True)
def get_equity_vector_index(community_cards: np.array):
    """
    Given the first three community cards and some a vector index, assign the value to
    a ncr(52, 4) vector that stores what rank the card is
    """
    pass

