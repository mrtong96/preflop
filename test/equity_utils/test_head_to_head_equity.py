import unittest

import numpy as np

from src.card_utils.card import get_card
from src.equity_utils.head_to_head_equity import get_head_to_head_equity


class TestHeadToHeadEquity(unittest.TestCase):
    def test_head_to_head_equity(self):
        expected_counts = np.array([391582, 1314513, 6209])

        shared_cards = []
        p1_cards = ['5C', '6C'] + shared_cards
        p2_cards = ['KD', 'KH'] + shared_cards
        p1_cards = np.array([get_card(card) for card in p1_cards], dtype=np.int8)
        p2_cards = np.array([get_card(card) for card in p2_cards], dtype=np.int8)

        equity = get_head_to_head_equity(p1_cards, p2_cards)

        self.assertTrue(np.all(equity == expected_counts), (equity, expected_counts))
