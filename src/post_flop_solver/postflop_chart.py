from typing import Optional

import numpy as np

from src.card_utils.hole_cards import get_index, get_2d_index
from src.post_flop_solver.post_flop_utils import VECTOR_SIZE, get_cards_from_vector_index
from src.post_flop_solver.postflop_bet import PostFlopBet, PostFlopAction
from src.preflop_solver.bet import BetType


# similar to the pre-flop chart except way harder...
class PostflopChart:
    OOP_INDEX = 0
    IP_INDEX = 1
    # so raise -> reraise
    MAX_RAISES_BEFORE_ALL_IN_PER_STREET = 2

    def __init__(
            self,
            community_cards: np.array,
            bet_sizes: np.array,
            stack_size: float,
            ip_hole_card_priors: Optional[np.array] = None,
            oop_hole_card_priors: Optional[np.array] = None,
            ip_vector_priors: Optional[np.array] = None,
            oop_vector_priors: Optional[np.array] = None,
            raises_this_street: int = 0,
            ip_to_act=False,
            is_first_street_action=True,
            bet_sequence: Optional[list[PostFlopBet]] = None,
    ):
        """
        Constructor

        :param community_cards: community cards in post-flop
        :param bet_sizes: array of size 3 of money already put in pot. oop/ip/other
        :param stack_size: total stack size of players
        :param ip_hole_card_priors: priors of the ip player for the 169-size vector
        :param oop_hole_card_priors: priors of the oop player for the 169-size vector
        :param ip_vector_priors: priors of the ip player for the ncr(52, 2)-size vector
        :param oop_vector_priors: priors of the oop player for the ncr(52, 2)-size vector
        :param raises_this_street: number or raises that have happened this street
        :param ip_to_act: who is supposed to be next
        :param is_first_street_action: If it's the IP's first decision after the flop/turn/river
        :param bet_sequence: sequence of bets leading to this position
        """
        self.community_cards = community_cards
        self.bet_sizes = bet_sizes
        self.stack_size = stack_size
        self.raises_this_street = raises_this_street
        self.ip_to_act = ip_to_act
        self.is_first_street_action = is_first_street_action
        self.bet_sequence = bet_sequence[:] if bet_sequence is not None else []  # copy the list

        if ip_hole_card_priors is None and ip_vector_priors is None:
            raise RuntimeError('one of ip_hole_cards and ip_vector_pairs must be set')
        if oop_hole_card_priors is None and oop_vector_priors is None:
            raise RuntimeError('one of oop_hole_card_priors and oop_vector_priors must be set')

        if ip_vector_priors is not None:
            self.ip_vector_priors = ip_vector_priors
        else:
            self.ip_vector_priors = np.zeros(VECTOR_SIZE, dtype=np.float64)
            for i in range(VECTOR_SIZE):
                lower_card, upper_card = get_cards_from_vector_index(i)
                hole_card_index = get_2d_index(lower_card, upper_card)
                self.ip_vector_priors[i] = ip_hole_card_priors[hole_card_index]

        if oop_vector_priors is not None:
            self.oop_vector_priors = oop_vector_priors
        else:
            self.oop_vector_priors = np.zeros(VECTOR_SIZE, dtype=np.float64)
            for i in range(VECTOR_SIZE):
                lower_card, upper_card = get_cards_from_vector_index(i)
                hole_card_index = get_2d_index(lower_card, upper_card)
                self.oop_vector_priors[i] = oop_hole_card_priors[hole_card_index]

        self.pot_size = np.sum(self.bet_sizes)

        self.children_charts: list['PostflopChart'] = []
        # we can always call because
        self.valid_bets = [BetType.FOLD, BetType.CALL]
        self._init_children()

        # what should we do in this position given cards (j, k) initialize it to a random decision
        self.decision_chart = np.zeros((len(BetType), VECTOR_SIZE), dtype=np.float64)
        valid_bets = [bet.value for bet in self.valid_bets]
        self.decision_chart[valid_bets] = 1 / len(valid_bets)

    def _compute_raise_size(self):
        """
        TODO: revisit raise rule, raise is half pot round up
        """
        if self.raises_this_street >= self.MAX_RAISES_BEFORE_ALL_IN_PER_STREET:
            return self.stack_size

        raise_size = min(np.ceil(self.pot_size) / 2, self.stack_size)
        raise_size += np.max(self.bet_sizes[:2])
        if raise_size >= self.stack_size * 0.6:
            raise_size = self.stack_size
        return raise_size

    def _init_children(self):
        # calling
        if self.is_first_street_action:
            child_chart = PostflopChart(
                self.community_cards,
                self.bet_sizes,
                self.stack_size,
                ip_vector_priors=self.ip_vector_priors,
                oop_vector_priors=self.oop_vector_priors,
                raises_this_street=self.raises_this_street,
                ip_to_act = not self.ip_to_act,
                is_first_street_action = False,
                bet_sequence = self.bet_sequence + [PostFlopBet(self.ip_to_act, PostFlopAction.CALL, self, None)]
            )
            self.children_charts.append(child_chart)
        # we check out the next card
        elif len(self.community_cards) < 5:
            new_bet_sizes = self.bet_sizes.copy()
            new_bet_sizes[:2] = np.max(new_bet_sizes[:2])

            for card in range(52):
                # no repeats
                if card in self.community_cards:
                    continue
                card_array = np.array(np.concatenate((self.community_cards, [card])), dtype=np.int8)
                child_chart = PostflopChart(
                    card_array,
                    new_bet_sizes,
                    self.stack_size,
                    ip_vector_priors=self.ip_vector_priors,
                    oop_vector_priors=self.oop_vector_priors,
                    raises_this_street=0,
                    ip_to_act=True,
                    is_first_street_action=True,
                    bet_sequence = self.bet_sequence + [PostFlopBet(self.ip_to_act, PostFlopAction.CALL, self, np.int8(card))]
                )
                self.children_charts.append(child_chart)

        # raise

        # can't raise if at max bet size
        if np.max(self.bet_sizes[:2]) == self.stack_size:
            return

        self.valid_bets.append(BetType.RAISE)
        raise_size = self._compute_raise_size()
        new_bet_sizes = self.bet_sizes.copy()
        raise_index = 1 if self.ip_to_act else 0
        new_bet_sizes[raise_index] = raise_size
        child_chart = PostflopChart(
            self.community_cards[:],
            new_bet_sizes,
            self.stack_size,
            ip_vector_priors=self.ip_vector_priors,
            oop_vector_priors=self.oop_vector_priors,
            raises_this_street=self.raises_this_street + 1,
            ip_to_act=not self.ip_to_act,
            is_first_street_action=False,
            bet_sequence=self.bet_sequence + [PostFlopBet(self.ip_to_act, PostFlopAction.RAISE, self, None)]
        )
        self.children_charts.append(child_chart)
