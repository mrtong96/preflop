from enum import Enum
from typing import Optional

import numpy as np


class PostFlopAction(Enum):
    # for now keep the bets as Enums...
    # not the most flexible but stupid enough to work for now
    FOLD = 0
    CALL = 1
    RAISE = 2

class PostFlopBet:
    """
    To serve as a history of what happened for charts

    Do Chart type hinting to prevent circular dependencies
    """
    def __init__(self, ip_to_act: bool, bet: PostFlopAction, chart: 'PostFlopChart', new_card: Optional[np.int8] = None):
        self.ip_to_act = ip_to_act
        self.bet_type = bet
        self.chart = chart
        self.new_card=new_card

    def __repr__(self):
        return f'({self.ip_to_act}, {self.bet_type.value}, {self.new_card})'