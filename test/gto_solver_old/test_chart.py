import unittest

import numpy as np

from src.card_utils.card import get_rank, RANKS, SUITS, get_suit
from src.gto_solver_old.chart import Chart


class TestCard(unittest.TestCase):
    def test_chart_initialization(self):
        bet_sizes = [1, 3.5, 12, 25]
        test_chart = Chart(0, 1, 1.5, bet_sizes,
                           0, 0)

        # we should have 5 bets initially
        self.assertEqual(len(test_chart.bet_sizes), len(bet_sizes) + 1)
        # probability of staying in each state is 1/5 to start with
        self.assertTrue(np.isclose(np.min(test_chart.conditional_bet_chances), np.max(test_chart.conditional_bet_chances)))
        self.assertTrue(np.isclose(np.min(test_chart.conditional_bet_chances), 1 / (len(bet_sizes) + 1)))

        probability_sums = np.sum(test_chart.conditional_bet_chances, axis=0)
        self.assertTrue(np.all(np.isclose(probability_sums, np.ones(probability_sums.shape))))
