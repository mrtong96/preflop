from enum import Enum

import numpy as np

DEFAULT_BET_SIZES = [3.5, 12.5, 37.5, 100]

class BetWinner(Enum):
    '''
    Who wins if the IP/OOP players decide to
    fold/raise/3-bet/etc
    '''
    IP_WINS = 0
    OOP_WINS = 1
    NEEDS_EQUITY = 2

'''
row: IP decision
col: OOP decision
val: result

(0, 0) -> OOP folds
(1+, 1+) -> resolves to same bet
(1+, 0) -> OOP folds
(0, 1+) -> IP folds
(1, 2+) -> OOP raise, hero call
(2, 1) -> OOP raise, IP 3-bet, OOP fold
(2, 3+) -> IP 3-bet, OOP 4-bet, IP fold
(3, 1) -> OOP raise, IP 3-bet, OOP fold
(3, 2) -> OOP raise, IP 3-bet, OOP call
(3, 4) -> OOP raise, IP 3-bet, OOP 4-bet, IP call
(4, 1) -> OOP raise, IP 3-bet, OOP fold
(4, 2) -> OOP raise, IP 3-bet, OOP call
(4, 3) -> OOP raise, IP 3-bet, OOP 4-bet, IP 5-bet, OOP fold
'''

BET_RESOLUTIONS = np.array([
    [0,1,1,1,1],
    [0,2,2,2,2],
    [0,0,2,1,1],
    [0,0,2,2,2],
    [0,0,2,0,2],
])

def get_bet_resolution(ip_bet_index, oop_bet_index):
    return BET_RESOLUTIONS[ip_bet_index, oop_bet_index]

def check_hero_wins(hero_bet_index: int, villain_bet_index: int, hero_in_position: bool) -> bool:
    """
    whether the hero wins without even needing to see the cards
    """
    ip_bet_index = hero_bet_index if hero_in_position else villain_bet_index
    oop_bet_index = villain_bet_index if hero_in_position else hero_bet_index

    bet_resolution_value = get_bet_resolution(ip_bet_index, oop_bet_index)
    if bet_resolution_value == BetWinner.NEEDS_EQUITY.value:
        return False
    elif hero_in_position and bet_resolution_value == BetWinner.IP_WINS.value:
        return True
    elif not hero_in_position and bet_resolution_value == BetWinner.OOP_WINS.value:
        return True
    else:
        return False