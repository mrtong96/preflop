# utils for handling cards
from numba import jit
import numpy as np

RANKS = '23456789TJQKA'
SUITS = 'SDCH'

@jit(nopython=True)
def get_rank(card: np.int8) -> np.int8:
    return card >> np.int8(2)

@jit(nopython=True)
def get_suit(card: np.int8) -> np.int8:
    return card & np.int8(3)

@jit(nopython=True)
def get_rank_array(cards: np.array) -> np.array:
    return cards >> np.int8(2)

@jit(nopython=True)
def get_suit_array(cards: np.array) -> np.array:
    return cards & np.int8(3)

def get_card(card: str) -> np.int8:
    return np.int8((RANKS.index(card[0]) << 2) | SUITS.index(card[1]))

def get_card_str(card: np.int8) -> str:
    return f'{RANKS[get_rank(card)]}{SUITS[get_suit(card)]}'
