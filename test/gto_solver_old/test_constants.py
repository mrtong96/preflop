import unittest

from src.gto_solver_old.constants import BET_RESOLUTIONS, BetWinner, check_hero_wins


class TestCard(unittest.TestCase):
    def test_check_hero_wins(self):
        for ip_bet in range(5):
            for oop_bet in range(5):
                # if we need equity nobody wins
                if BET_RESOLUTIONS[ip_bet, oop_bet] == BetWinner.NEEDS_EQUITY.value:
                    self.assertFalse(check_hero_wins(ip_bet, oop_bet, True), (ip_bet, oop_bet))
                    self.assertFalse(check_hero_wins(oop_bet, ip_bet, False), (ip_bet, oop_bet))
                else:
                    # someone wins when a win occurs
                    ip_wins = check_hero_wins(ip_bet, oop_bet, True)
                    oop_wins = check_hero_wins(oop_bet, ip_bet, False)

                    self.assertTrue(ip_wins ^ oop_wins, (ip_bet, oop_bet, ip_wins, oop_wins))