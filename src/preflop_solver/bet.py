from enum import Enum
from typing import Optional


# for now keep the bets as Enums...
# not the most flexible but stupid enough to work for now
class BetType(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2

class Bet:
    """
    To serve as a history of what happened for charts

    Do Chart type hinting to prevent circular dependencies
    """
    def __init__(self, position: int, bet_type: BetType, chart: 'Chart'):
        self.position = position
        self.bet_type = bet_type
        self.chart = chart

    def __repr__(self):
        return f'({self.position}, {self.bet_type.value})'