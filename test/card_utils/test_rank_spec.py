import unittest

import numpy as np

from src.card_utils.card import get_card, RANKS
from src.card_utils.combinatorics_utils import ncr
from src.card_utils.constants import STRAIGHT_FLUSH, FULL_HOUSE, STRAIGHT, FOUR_KIND, FLUSH, THREE_KIND, TWO_PAIR, ONE_PAIR
from src.card_utils.rank_spec import RankSpec

class TestRankSpec(unittest.TestCase):
    def test_straight(self):
        test_cases = [
            # ace is low
            [[12, 0, 1, 2, 3], 3],
            # ace is high
            [[8, 9 , 10, 11, 12], 12],
            # long straight
            [[2,3,4,5,6,7,8], 8],
            # no
            [[10, 11, 12, 0, 1], None],
            [[0, 1, 4, 5, 6], None],
        ]

        for cards, expected_straight in test_cases:
            empty_cards = np.array([], dtype=np.int8)
            community_cards = np.array([card << 2 for card in cards], dtype=np.int8)
            rank_spec = RankSpec(empty_cards, empty_cards, community_cards)

            self.assertEqual(rank_spec.is_straight, expected_straight is not None, (cards, expected_straight))
            if expected_straight is not None:
                self.assertEqual(rank_spec.get_straight_rank(rank_spec.rank_counts), expected_straight)

    def test_straight_flush(self):
        test_cases = [
            [['3S', '4S', 'TH', 'JH', 'QH', 'KH', 'AH'], True],
            # straight and flush
            [['8H', '9H', 'TS', 'JH', 'QH', 'KS', 'AH'], False],
            # huh?

            [['KH', 'KC', '9H', 'TH', 'JH', 'QH', 'AS'], True]
        ]

        for cards, expected in test_cases:
            cards = np.array([get_card(card) for card in cards], dtype=np.int8)
            rank_spec = RankSpec(cards[:2], cards[:2], cards[2:])

            self.assertEqual(rank_spec.hand_is_straight_flush(), expected,
                             (cards, expected,
                              rank_spec.is_flush, rank_spec.is_straight,
                              rank_spec.get_flush_rank_counts(),
                              np.where(rank_spec.suit_counts >= 5)[0][0]))

    def test_iter_count(self):
        deck = np.array(list(range(52)), dtype=np.int8)

        rank_spec = RankSpec(deck[:2], deck[2:4], deck[4:9])

        rotations = 1
        while rank_spec.attempt_rotation():
            rotations += 1
        self.assertEqual(rotations, ncr(48, 5))

        rank_spec = RankSpec(deck[0:0], deck[0:0], deck[0:5])
        rotations = 1
        while rank_spec.attempt_rotation():
            rotations += 1
        self.assertEqual(rotations, ncr(52, 5))

        rank_spec = RankSpec(deck[0:5], deck[5:7], deck[7:9])
        rotations = 1
        while rank_spec.attempt_rotation():
            rotations += 1
        self.assertEqual(rotations, ncr(52 - 7, 2))

    def test_combo_counts(self):
        """
        Test some 5-card combos to assert that the counts are correct
        """
        straight_flush_count = 0
        four_kind_count = 0
        full_house_count = 0
        flush_count = 0
        straight_count = 0
        three_kind_count = 0
        two_pair_count = 0
        one_pair_count = 0
        iterations = 0

        deck = np.array(list(range(52)), dtype=np.int8)
        rank_spec = RankSpec(deck[0:0], deck[0:0], deck[0:5])

        while True:
            rank = rank_spec.get_hand_rank()

            if rank >> 12 == ((STRAIGHT_FLUSH << 1) + 1):
                straight_flush_count += 1
            elif rank >> 12 == (FOUR_KIND << 1):
                four_kind_count += 1
            elif rank >> 13 == FULL_HOUSE:
                full_house_count += 1
            elif rank >> 13 == FLUSH:
                flush_count += 1
            elif rank >> 13 == STRAIGHT:
                straight_count += 1
            elif rank >> 13 == THREE_KIND:
                three_kind_count += 1
            elif rank >> 13 == TWO_PAIR:
                two_pair_count += 1
            elif rank >> 13 == ONE_PAIR:
                self.assertNotEqual(rank & 0x1FF, 0, (
                    "kicker number is 0",
                    rank_spec.community_cards,
                    rank_spec.rank_counts,
                    rank_spec.arg_sort_rank_counts,
                ))

                one_pair_count += 1

            iterations += 1
            if not rank_spec.attempt_rotation():
                break

        self.assertEqual(straight_flush_count, 4 * 10)
        self.assertEqual(four_kind_count, 13 * (52 - 4))
        self.assertEqual(full_house_count, 13 * 12 * ncr(4, 3) * ncr(4, 2))
        self.assertEqual(flush_count, 4 * ncr(13, 5) - straight_flush_count)
        self.assertEqual(straight_count, (10 * (4 ** 5)) - straight_flush_count)
        self.assertEqual(three_kind_count, (13 * ncr(4, 3) * ncr(48, 2)) - full_house_count)
        self.assertEqual(two_pair_count, ncr(13, 2) * ncr(4, 2) * ncr(4, 2) * (52 - 8))
        self.assertEqual(one_pair_count, 13 * ncr(4, 2) * (52 - 4) * (52 - 8) * (52 - 12) / 6)
        self.assertEqual(iterations, ncr(52, 5))

    def test_two_pair_rank(self):
        def build_two_pair_rank(high_pair, low_pair, kicker):
            return np.uint16(
                (TWO_PAIR << 13) |
                (RANKS.index(high_pair) << 8) |
                (RANKS.index(low_pair) << 4) |
                (RANKS.index(kicker))
            )

        test_cases = [
            [['3S', '3H', '4H', '4D', '5D', '5C', 'AD'], '54A'],
            [['AS', 'AH', 'KH', 'KD', '2D', '4C', '7D'], 'AK7'],
            [['AS', 'AH', 'KH', 'KD', 'QD', 'QC', '7D'], 'AKQ'],
        ]
        empty_array = np.array([], dtype=np.int8)

        for cards, expected in test_cases:
            expected_rank = build_two_pair_rank(*expected)
            cards = np.array([get_card(card) for card in cards], dtype=np.int8)
            rank_spec = RankSpec(empty_array, empty_array, cards)
            self.assertEqual(expected_rank, rank_spec.get_hand_rank())

    def test_three_kind_rank(self):
        def build_three_pair_rank(three_kind, high_kicker, low_kicker):
            return np.uint16(
                (THREE_KIND << 13) |
                (RANKS.index(three_kind) << 8) |
                (RANKS.index(high_kicker) << 4) |
                (RANKS.index(low_kicker))
            )

        test_cases = [
            [['3S', '3H', '3C', '4D', '5S', '7C', 'TD'], '3T7'],
            [['AS', 'AH', 'AD', 'KD', 'QC', '4C', '7D'], 'AKQ'],
        ]
        empty_array = np.array([], dtype=np.int8)

        for cards, expected in test_cases:
            expected_rank = build_three_pair_rank(*expected)
            cards = np.array([get_card(card) for card in cards], dtype=np.int8)
            rank_spec = RankSpec(empty_array, empty_array, cards)
            self.assertEqual(expected_rank, rank_spec.get_hand_rank())
