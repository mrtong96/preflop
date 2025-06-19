import numba
import numpy as np
from numba.experimental import jitclass

from src.card_utils.card import get_rank, get_suit, RANKS, SUITS
from src.card_utils.constants import STRAIGHT_FLUSH, FOUR_KIND, FULL_HOUSE, FLUSH, STRAIGHT, THREE_KIND, TWO_PAIR, \
    ONE_PAIR, THREE_CARD_TO_CODE

FIVE_INDEX = RANKS.index('5')
TEN_INDEX = RANKS.index('T')

rank_spec = [
    ('fixed_cards', numba.int8[:]),
    ('block_cards', numba.int8[:]),
    ('community_cards', numba.int8[:]),
    ('card_vector', numba.int8[:]),
    ('rank_vector', numba.int8[:]),
    ('suit_vector', numba.int8[:]),
    ('rank_counts', numba.int8[:]),
    ('suit_counts', numba.int8[:]),
    ('arg_sort_rank_counts', numba.int8[:]),
    ('is_flush', numba.bool_),
    ('is_straight', numba.bool_),
    ('hand_rank', numba.uint16),
    ('use_cached_rank', numba.bool_),
]


@jitclass(rank_spec)
class RankSpec(object):
    """
    Code to compute all the possible community cards and rank them really fast.
    Currently, it can score about 10M hands per second or about 5-6 equity combos per second
    """
    def __init__(self, hole_cards: np.array, block_cards: np.array, community_cards: np.array):
        """
        build the rank spec

        :param hole_cards: the two hole cards for the hand
        :param block_cards: other cards that you know are out
        :param community_cards: the five shared cards, they should be sorted
        """
        self.fixed_cards = np.array([card for card in hole_cards], dtype=np.int8)
        self.block_cards = np.zeros(len(hole_cards) + len(block_cards), dtype=np.int8)
        self.block_cards[:len(hole_cards)] = self.fixed_cards
        self.block_cards[len(hole_cards):] = np.array([card for card in block_cards], dtype=np.int8)
        self.community_cards = np.array([card for card in community_cards], dtype=np.int8)
        self.card_vector = np.zeros(len(hole_cards) + len(community_cards), dtype=np.int8)
        self.card_vector[:len(hole_cards)] = self.fixed_cards
        self.card_vector[len(hole_cards):] = self.community_cards
        self.rank_vector = np.array([get_rank(card) for card in self.card_vector], dtype=np.int8)
        self.suit_vector = np.array([get_suit(card) for card in self.card_vector], dtype=np.int8)
        self.rank_counts = self.get_rank_counts()
        self.suit_counts = self.get_suit_counts()
        arg_sort = np.argsort(self.rank_counts)[::-1]
        self.arg_sort_rank_counts = np.array([el for el in arg_sort], dtype=np.int8)

        self.is_flush = np.max(self.suit_counts) >= 5
        self.is_straight = self.hand_is_straight(self.rank_counts)

        self.hand_rank = np.uint16(0)
        self.use_cached_rank = False

    def get_rank_counts(self) -> np.array:
        """
        get the rank counts of the vector
        """
        rank_counts = np.zeros(len(RANKS), dtype=np.int8)
        for rank in self.rank_vector:
            rank_counts[rank] += 1
        return rank_counts

    def get_suit_counts(self) -> np.array:
        """
        get the rank counts of the vector
        """
        suit_counts = np.zeros(len(SUITS), dtype=np.int8)
        for suit in self.suit_vector:
            suit_counts[suit] += 1
        return suit_counts

    @staticmethod
    def hand_is_straight(rank_counts) -> bool:
        # can't have a straight if you are missing 5T
        if rank_counts[FIVE_INDEX] == 0 and rank_counts[TEN_INDEX] == 0:
            return False

        start_rank = -1
        while start_rank < len(RANKS) - 4:
            for increment in range(5):
                if rank_counts[start_rank + increment] == 0:
                    start_rank += increment + 1
                    break
            else:
                return True
        return False

    def get_flush_rank_counts(self) -> np.array:
        flush_suit = np.where(self.suit_counts >= 5)[0][0]
        flush_rank_counts = np.zeros(len(RANKS), dtype=np.int8)
        for card in self.card_vector:
            if get_suit(card) == flush_suit:
                flush_rank_counts[get_rank(card)] += 1
        return flush_rank_counts

    def hand_is_straight_flush(self) -> bool:
        if not self.is_flush or not self.is_straight:
            return False
        flush_rank_counts = self.get_flush_rank_counts()
        return self.hand_is_straight(flush_rank_counts)

    def replace_card(self, add_card: np.int8, community_card_index: int):
        # need the index of the card to remove
        remove_card = np.int8(self.community_cards[community_card_index])
        card_index = community_card_index + len(self.fixed_cards)

        # compute the rank/suit of the cards to add/remove
        add_rank = get_rank(add_card)
        add_suit = get_suit(add_card)
        remove_rank = get_rank(remove_card)
        remove_suit = get_suit(remove_card)

        # we remove our flush
        if self.is_flush and self.suit_counts[remove_suit] == 5:
            self.is_flush = False
            self.use_cached_rank = False

        # state variable updates
        self.community_cards[community_card_index] = add_card
        self.card_vector[card_index] = add_card
        self.rank_vector[card_index] = add_rank
        self.suit_vector[card_index] = add_suit
        self.suit_counts[add_suit] += 1
        self.suit_counts[remove_suit] -= 1

        # we have a flush
        if self.suit_counts[add_suit] >= 5:
            self.is_flush = True
            self.use_cached_rank = False

        # rank-related operations are really expensive, only compute if ranks change
        if add_rank != remove_rank:
            self.rank_counts[add_rank] += 1
            self.rank_counts[remove_rank] -= 1
            self.update_argsort(add_rank, remove_rank)

            self.is_straight = self.hand_is_straight(self.rank_counts)
            self.use_cached_rank = False


    # custom method to update an argsort result because you know there are
    # only two indexes to update instead of all 13
    def update_argsort(self, add_rank, remove_rank):
        # remove a rank
        for index in range(len(RANKS) - 1):
            if self.arg_sort_rank_counts[index] == remove_rank:
                current_count = self.rank_counts[self.arg_sort_rank_counts[index]]
                lower_count = self.rank_counts[self.arg_sort_rank_counts[index + 1]]

                # swap
                if current_count < lower_count:
                    tmp = self.arg_sort_rank_counts[index+1]
                    self.arg_sort_rank_counts[index+1] = self.arg_sort_rank_counts[index]
                    self.arg_sort_rank_counts[index] = tmp
                else:
                    break

        # add a rank
        for index in range(len(RANKS) - 1, 0, -1):
            if self.arg_sort_rank_counts[index] == add_rank:
                current_count = self.rank_counts[self.arg_sort_rank_counts[index]]
                higher_count = self.rank_counts[self.arg_sort_rank_counts[index - 1]]

                # swap
                if current_count > higher_count:
                    tmp = self.arg_sort_rank_counts[index-1]
                    self.arg_sort_rank_counts[index - 1] = self.arg_sort_rank_counts[index]
                    self.arg_sort_rank_counts[index] = tmp
                else:
                    break

    def attempt_rotation(self) -> bool:
        """
        iterator to rotate stuff

        :return: True if can rotate, false if can not
        """
        offset = 0
        card_index = 0
        while card_index < len(self.community_cards):
            community_card = self.community_cards[-card_index - 1]
            if 52 - card_index - offset - 1 in self.block_cards:
                offset += 1
            elif community_card != 52 - card_index - offset - 1:
                break
            else:
                card_index += 1

        # can not rotate anymore. At the end
        if card_index == len(self.community_cards):
            return False

        # first card to add
        first_index = len(self.community_cards) - 1 - card_index
        add_card = 1 + self.community_cards[first_index]
        while add_card in self.block_cards:
            add_card += 1
        self.replace_card(add_card, first_index)

        for next_index in range(first_index + 1, len(self.community_cards)):
            next_add_card = 1 + add_card
            while next_add_card in self.block_cards:
                next_add_card += 1

            self.replace_card(next_add_card, next_index)
            add_card = next_add_card

        return True

    def get_hand_rank(self) -> np.uint16:
        if self.use_cached_rank:
            return self.hand_rank

        if self.hand_is_straight_flush():
            self.hand_rank = self.get_straight_flush_rank_int()
        elif self.rank_counts[self.arg_sort_rank_counts[0]] == 4:
            self.hand_rank = self.get_four_kind_rank_int()
        elif self.rank_counts[self.arg_sort_rank_counts[0]] == 3 and self.rank_counts[self.arg_sort_rank_counts[1]] >= 2:
            self.hand_rank = self.get_full_house_rank_int()
        elif self.is_flush:
            self.hand_rank = self.get_flush_rank_int()
        elif self.is_straight:
            self.hand_rank = self.get_straight_rank_int()
        elif self.rank_counts[self.arg_sort_rank_counts[0]] == 3:
            self.hand_rank = self.get_three_kind_rank_int()
        # two pair
        elif self.rank_counts[self.arg_sort_rank_counts[1]] == 2:
            self.hand_rank = self.get_two_pair_rank_int()
        elif self.rank_counts[self.arg_sort_rank_counts[0]] == 2:
            self.hand_rank = self.get_one_pair_rank_int()
        # high card
        else:
            self.hand_rank = self.get_high_card_rank_int()

        self.use_cached_rank = True
        return self.hand_rank

    def get_straight_flush_rank_int(self) -> np.uint16:
        flush_rank_counts = self.get_flush_rank_counts()
        straight_rank = self.get_straight_rank(flush_rank_counts)

        return np.uint16(
            (STRAIGHT_FLUSH << 13) |
            (1 << 12) |
            straight_rank
        )

    def get_straight_rank(self, rank_counts) -> np.uint16:
        for start_rank in range(len(RANKS) - 1, 2, -1):
            found_straight = True
            for increment in range(5):
                if rank_counts[start_rank - increment] == 0:
                    found_straight = False
                    break
            if found_straight:
                return np.uint16(start_rank)

        raise RuntimeError("should have found a straight but did not: " + str(len(self.rank_counts)))

    def get_four_kind_rank_int(self) -> np.uint16:
        four_kind_rank = np.int8(self.arg_sort_rank_counts[0])

        kicker = -1
        index = 12
        while index >= 0:
            if self.rank_counts[index] > 0 and index != four_kind_rank:
                kicker = index
                break
            index -= 1

        if int(four_kind_rank) == -1 or int(kicker) == -1:
            raise RuntimeError(f"-1 found: {self.community_cards}")

        return np.uint16(
            (FOUR_KIND << 13) |
            (four_kind_rank << 4) |
            kicker
        )

    def get_full_house_rank_int(self) -> np.uint16:
        three_kind_rank = -1
        index = 12
        while index >= 0:
            if self.rank_counts[index] == 3:
                three_kind_rank = index
                break
            index -= 1

        kicker = -1
        index = 12
        while index >= 0:
            if index != three_kind_rank and self.rank_counts[index] >= 2:
                kicker = index
                break
            index -= 1

        return np.uint16(
            (FULL_HOUSE << 13) |
            (three_kind_rank << 4) |
            kicker
        )

    def get_flush_rank_int(self) -> np.uint16:
        flush_suit = np.int8(np.where(self.suit_counts >= 5)[0][0])
        # slow but doesn't run too often since there aren't that many flushes
        sorted_cards = np.sort(self.card_vector)

        rank_int = np.uint16(0)
        cards_found = 0
        for card in sorted_cards[::-1]:
            if get_suit(card) == flush_suit:
                rank_int |= (1 << get_rank(card))
                cards_found += 1
                if cards_found == 5:
                    break

        return np.uint16(
            (FLUSH << 13) |
            rank_int
        )

    def get_straight_rank_int(self) -> np.uint16:
        return np.uint16(
            (STRAIGHT << 13) |
            self.get_straight_rank(self.rank_counts)
        )

    def get_three_kind_rank_int(self) -> np.uint16:
        three_kind_rank = np.uint16(self.arg_sort_rank_counts[0])

        low_kicker = -1
        high_kicker = -1
        index = 12
        while index >= 0:
            if self.rank_counts[index] > 0 and index != three_kind_rank:
                # hack because there are only two kicker cards
                if high_kicker == -1:
                    high_kicker = index
                else:
                    low_kicker = index
                    break
            index -= 1

        return np.uint16(
            (THREE_KIND << 13) |
            (three_kind_rank << 8) |
            (high_kicker << 4) |
            low_kicker
        )

    def get_two_pair_rank_int(self) -> np.uint16:
        index = 12
        low_pair = -1
        high_pair = -1
        while index >= 0:
            if self.rank_counts[index] == 2:
                # hack because there are only two kicker cards
                if high_pair == -1:
                    high_pair = index
                else:
                    low_pair = index
                    break
            index -= 1

        index = 12
        kicker = -1
        while index >= 0:
            if self.rank_counts[index] > 0 and index != high_pair and index != low_pair:
                kicker = index
                break
            index -= 1

        return np.uint16(
            (TWO_PAIR << 13) |
            (high_pair << 8) |
            (low_pair << 4) |
            kicker
        )

    def get_one_pair_rank_int(self) -> np.uint16:
        pair_rank = np.int8(self.arg_sort_rank_counts[0])

        index = 12
        high_kicker = -1
        mid_kicker = -1
        low_kicker = -1

        while index >= 0:
            if index != pair_rank and self.rank_counts[index] > 0:
                if high_kicker == -1:
                    high_kicker = index
                elif mid_kicker == -1:
                    mid_kicker = index
                else:
                    low_kicker = index
                    break
            index -= 1

        return np.uint16(
            (ONE_PAIR << 13) |
            (pair_rank << 9) |
            THREE_CARD_TO_CODE[high_kicker, mid_kicker, low_kicker]
        )

    def get_high_card_rank_int(self) -> np.uint16:
        rank_int = np.uint16(0)

        cards_found = 0
        index = 12
        while True:
            if self.rank_counts[index] > 0:
                rank_int |= (1 << index)
                cards_found += 1
                if cards_found == 5:
                    break
            index -= 1

        return rank_int
