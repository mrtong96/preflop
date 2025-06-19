import unittest

import numpy as np

from src.card_utils.card import get_rank, RANKS, SUITS, get_suit


class TestCard(unittest.TestCase):
    def test_rank(self):
        for rank in range(len(RANKS)):
            for suit in range(len(SUITS)):
                card = np.int8(4 * rank + suit)
                self.assertEqual(get_rank(card), rank)
                self.assertEqual(get_suit(card), suit)
