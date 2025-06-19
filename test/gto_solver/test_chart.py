import unittest

import numpy as np

from src.preflop_solver.bet import BetType
from src.preflop_solver.chart import Chart


class TestCard(unittest.TestCase):
    @staticmethod
    def _chart_tree_to_list(chart):
        chart_list = [chart]
        for child_chart in chart.children_charts:
            chart_list.extend(TestCard._chart_tree_to_list(child_chart))
        return chart_list

    def test_chart_construction(self):
        """
        Some basic sanity checks that the chart tree behavior looks as expected
        """

        num_players = 6
        bets = np.zeros(num_players, dtype=np.float64)
        bets[-2] = 0.5
        bets[-1] = 1.0
        is_in = np.ones(num_players, dtype=np.bool_)
        vpip = np.zeros(num_players, dtype=np.bool_)

        root_chart = Chart(
            hero_position=0,
            bets=bets,
            is_in=is_in,
            vpip=vpip,
            max_vpip_players=2,
            players=num_players,
            stack_size=100,
        )
        chart_list = TestCard._chart_tree_to_list(root_chart)

        for chart in chart_list:
            # Check the calling behavior
            call_bets = [bet for bet in chart.bet_sequence if bet.bet_type == BetType.CALL]
            check = len(call_bets) == 0 or (len(call_bets) == 1 and call_bets[0].position == call_bets[0].chart.players - 2)
            msg = f"call should result in terminating the sequence with max_vpip_players=2 or be from SB, {chart.bet_sequence}"
            self.assertTrue(check, msg)

            # with these settings the max we can do is 5-bet, there can only be 4 raises
            raise_bets = [bet for bet in chart.bet_sequence if bet.bet_type == BetType.RAISE]
            check = len(raise_bets) <= 4
            msg = f"with these settings you should not be able to 6+-bet, {chart.bet_sequence}"
            self.assertTrue(check, msg)

            # if you folded you can not suddenly call or raise
            folded_players = set()
            for bet in chart.bet_sequence:
                if bet.bet_type == BetType.FOLD:
                    folded_players.add(bet.position)
                else:
                    self.assertTrue(bet.position not in folded_players, "player folded then called/raised")

            # each chart can't have more than len(BetType) different children (i.e. number of possible decisions)
            self.assertTrue(len(chart.children_charts) <= len(BetType))