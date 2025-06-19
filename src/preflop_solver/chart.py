import logging
from typing import Optional

import numpy as np

from src.equity_utils.joint_equity_probabilities import load_conditional_joint_probabilities, \
    compute_weighted_equity_vector
from src.preflop_solver.bet import BetType, Bet


class Chart:
    """
    Main class that we want to use to represent betting strategy.
    Each chart is a what-if position that we need to slowly converge to a GTO solution.
    """
    def __init__(self, hero_position: int,
                 bets: np.array, is_in: np.array, vpip: np.array,
                 max_vpip_players=2, players=6, stack_size=100,
                 rake=0.05,
                 bet_sequence: Optional[list[Bet]]=None, no_limping=True,):
        """
        Constructor

        :param hero_position: integer in range [0, players), 0 is UTG, players-1 is BB
        :param bets: array of bets indexed by player position
        :param is_in: array of booleans indicating who has not folded yet
        :param vpip: array of booleans indicating who has voluntarily put money in
        :param max_vpip_players: multi pots are hard. Most number of people that voluntarily put money in to
            consider, we assume everyone else folds if we are at the max vpip bets
        :param players: total number of players at the table
        :param stack_size: maximum bet size that we can have
        :param rake: rake rate when winning
        :param bet_sequence: children charts.
        :param no_limping: no limping if not in SB position
        """
        self.hero_position = hero_position
        self.bets = bets
        self.is_in = is_in
        self.vpip = vpip
        self.maximum_vpip_players = max_vpip_players
        self.players = players
        self.stack_size = stack_size
        self.rake = rake
        self.bet_sequence = bet_sequence[:] if bet_sequence is not None else []  # copy the list
        self.no_limping = no_limping

        self.validate_params()

        self.pot_size = np.sum(bets)
        self.no_fold_bet_sequence = [bet for bet in self.bet_sequence if bet.bet_type != BetType.FOLD]

        # we want to maintain a linked list of children.
        # self.parent_charts: list['Chart'] = [bet.chart for bet in self.bet_sequence]
        self.children_charts: list['Chart'] = []
        self.valid_bets = []
        self._init_children()

        # what should we do in this position given cards (j, k) initialize it to a random decision
        self.decision_chart = np.zeros((len(BetType), 169), dtype=np.float64)
        valid_bets = [bet.value for bet in self.valid_bets]
        self.decision_chart[valid_bets] = 1 / len(valid_bets)

    def validate_params(self):
        if not self.is_in[self.hero_position]:
            raise RuntimeError("we shouldn't be in this hand according to self.is_in")

        if np.sum(self.is_in) <= 1:
            raise RuntimeError("Nobody else is in this hand")

        if self.maximum_vpip_players > 2:
            logging.warning("do not support multi-pot equilibriums computations")

    def _get_next_hero_position(self):
        next_hero_position = self.hero_position
        while True:
            next_hero_position = (next_hero_position + 1) % self.players
            if self.is_in[next_hero_position]:
                break

        return next_hero_position

    def _compute_raise_size(self):
        """
        TODO: revisit raise rule, basically exponentially increase pot size and shove if too close to all in
        """
        raise_size = self.pot_size * 2 + 0.5
        if raise_size >= self.stack_size * 0.6:
            raise_size = self.stack_size
        return raise_size

    def _init_child(self, next_hero_position, new_bet, new_bets=None, new_is_in=None, new_vpip=None):
        return Chart(
            hero_position=next_hero_position,
            bets=new_bets if new_bets is not None else self.bets.copy(),
            is_in=new_is_in if new_is_in is not None else self.is_in.copy(),
            vpip=new_vpip if new_vpip is not None else self.vpip.copy(),
            max_vpip_players=self.maximum_vpip_players,
            players=self.players,
            stack_size=self.stack_size,
            bet_sequence=self.bet_sequence[:] + [new_bet],
            rake=self.rake,
            no_limping=self.no_limping,
        )

    def _init_children(self):
        next_hero_position = self._get_next_hero_position()

        # folding
        fold_bet = Bet(self.hero_position, BetType.FOLD, self)
        fold_is_in = self.is_in.copy()
        fold_is_in[self.hero_position] = False
        self.valid_bets.append(BetType.FOLD)

        # if there is only one person left in the hand, we don't need to recurse
        if np.sum(fold_is_in) > 1:
            fold_chart = self._init_child(next_hero_position, fold_bet, new_is_in=fold_is_in)
            self.children_charts.append(fold_chart)

        # Too many vpip players and you haven't joined yet, only fold
        if np.sum(self.vpip) >= self.maximum_vpip_players and not self.vpip[self.hero_position]:
            return

        # calling
        call_bet = Bet(self.hero_position, BetType.CALL, self)
        call_vpip = self.vpip.copy()
        call_vpip[self.hero_position] = True
        call_bets = self.bets.copy()
        call_bets[self.hero_position] = np.max(self.bets)

        # kind of hacky but works. Also allows for small blind to do tricky play
        is_first_small_blind_bet = self.hero_position == self.players - 2 and len(self.bet_sequence) < self.players
        is_limping = call_bets[self.hero_position] == 1
        vpip_limit = all(
            [bet.bet_type == BetType.CALL for bet in self.no_fold_bet_sequence[:-(self.maximum_vpip_players - 1)]])
        # we did not hit the limit if we already put money in. Used for SB call edge case
        vpip_limit = vpip_limit and not self.vpip[self.hero_position]
        matches_next_bet = call_bets[self.hero_position] == self.bets[next_hero_position]

        # if we are allowed to call and we pass the limp check, or we are the first bet in small blind
        if (not vpip_limit and self.no_limping and not is_limping) or is_first_small_blind_bet:
            self.valid_bets.append(BetType.CALL)
            # if we call to match the next person's bet, then flop would happen, don't need to recurse
            if not matches_next_bet or is_first_small_blind_bet:
                call_chart = self._init_child(next_hero_position, call_bet, new_bets=call_bets, new_vpip=call_vpip)
                self.children_charts.append(call_chart)

        raise_bet = Bet(self.hero_position, BetType.RAISE, self)
        raise_vpip = self.vpip.copy()
        raise_vpip[self.hero_position] = True
        raise_bets = self.bets.copy()
        raise_size = self._compute_raise_size()
        raise_bets[self.hero_position] = raise_size

        # if we are in a position to raise
        if np.max(self.bets) < self.stack_size:
            self.valid_bets.append(BetType.RAISE)
            raise_chart = self._init_child(next_hero_position, raise_bet, new_bets=raise_bets, new_vpip=raise_vpip)
            self.children_charts.append(raise_chart)

    def _get_hypothetical_equity_vector(self, num_fixed_bets: int, fixed_hero_position: int,
                                        fixed_prior_range: np.array, fixed_hypothetical_bet_type: BetType,
                                        prior_ranges: np.array):
        """
        Estimate the equity for the fixed hero by traversing the rest of the tree... this is going to be hard
        """
        if len(self.bet_sequence) == num_fixed_bets:
            raise RuntimeError("something is wrong with the bet sequence length")

        # given fixed_hero has cards i, what is the probability that the opponent has cards j
        conditional_probabilities = load_conditional_joint_probabilities()
        # you know that from the priors that the current player has a range
        priors = prior_ranges[self.hero_position].reshape((1, -1))

        # if we have no probability of being here, who cares
        if np.sum(priors) == 0:
            return np.zeros(169, dtype=np.float64)

        conditional_probabilities = conditional_probabilities * priors
        # for every card i, we want the probability that we arrive at a decision
        # (169x169 dot 169x3).T -> 3x169 matrix of decision probabilities
        decision_probabilities = np.dot(conditional_probabilities, self.decision_chart.T).T

        # normalize
        decision_probabilities /= np.sum(decision_probabilities, axis=0)

        # compute the hard part (results matrix)
        equity_matrix = np.zeros((len(BetType), 169), dtype=np.float64)
        for valid_bet in self.valid_bets:
            next_chart = [chart for chart in self.children_charts if chart.bet_sequence[-1].bet_type == valid_bet]
            next_prior_ranges = prior_ranges.copy()
            next_prior_ranges[self.hero_position] *= self.decision_chart[valid_bet.value].ravel()

            # if the bet isn't settled, recurse
            if next_chart:
                equity_matrix[valid_bet.value] = next_chart[0]._get_hypothetical_equity_vector(
                    num_fixed_bets=num_fixed_bets, fixed_hero_position=fixed_hero_position,
                    fixed_prior_range=fixed_prior_range, fixed_hypothetical_bet_type=valid_bet,
                    prior_ranges=next_prior_ranges)
            else:
                if valid_bet == BetType.RAISE:
                    raise RuntimeError("we should always have something after a raise")
                elif valid_bet == BetType.FOLD:
                    # the fixed hero folds, negative equity
                    if self.hero_position == fixed_hero_position:
                        equity_matrix[valid_bet.value] = - self.bets[fixed_hero_position]
                    # the other guy folds, fixed hero wins
                    else:
                        equity_matrix[valid_bet.value] = (self.pot_size * (1 - self.rake)) - self.bets[fixed_hero_position]
                # call
                else:
                    # compute perform the equity calculation
                    if np.sum(self.is_in > self.maximum_vpip_players):
                        raise RuntimeError(f"too many players to evaluate, {self.is_in}, {self.maximum_vpip_players}")

                    equity_amount = self._get_call_equity_amount(fixed_hero_position, prior_ranges)
                    equity_matrix[valid_bet.value] = equity_amount

        # compute the results
        weighted_equity_matrix = equity_matrix * decision_probabilities
        return np.sum(weighted_equity_matrix, axis=0)

    def _get_hypothetical_equity_matrix(self, num_fixed_bets: int, fixed_hero_position: int, prior_ranges: np.array):
        """
        Compute the equity of a hypothetical. Tricky because uses graph traversal and my head hurts.
        TODO: rewrite this part once max_vpip_players > 2

        :param fixed_hero_position: the position of the person calling that is fixed for the purposes of equity calculations
        :param prior_ranges: the range of all the players
        """
        # we want to build an equity matrix and return it
        equity_matrix = np.zeros(shape=self.decision_chart.shape, dtype=np.float64)
        for valid_bet in self.valid_bets:
            # folding for the hypothetical is pretty easy, you don't care about how the other
            # players fight for the pot
            if valid_bet == BetType.FOLD:
                equity_matrix[valid_bet.value] = -self.bets[fixed_hero_position]
                continue

            # see if we have a chart to compute
            next_chart = [chart for chart in self.children_charts if chart.bet_sequence[-1].bet_type == valid_bet]

            # call the helper method with the next chart
            if next_chart:
                equity_matrix[valid_bet.value] = next_chart[0]._get_hypothetical_equity_vector(
                    num_fixed_bets=num_fixed_bets, fixed_hero_position=fixed_hero_position,
                    fixed_prior_range=prior_ranges[fixed_hero_position], fixed_hypothetical_bet_type=valid_bet,
                    prior_ranges=prior_ranges)

                continue

            # compute perform the equity calculation
            if np.sum(self.is_in > self.maximum_vpip_players):
                raise RuntimeError(f"too many players to evaluate, {self.is_in}, {self.maximum_vpip_players}")
            elif valid_bet != BetType.CALL:
                raise RuntimeError(f"should be evaluating equity here with calls only, {valid_bet}, {self.bet_sequence}, {self.hero_position}")
            elif fixed_hero_position != self.hero_position:
                raise RuntimeError(f"you should only call in hero position here")

            equity_amount = self._get_call_equity_amount(fixed_hero_position, prior_ranges)

            equity_matrix[valid_bet.value] = equity_amount

        return equity_matrix

    def _get_call_equity_amount(self, fixed_hero_position, prior_ranges):
        # this part really relies on self.max_vpip_players == 2
        villain_position = list(self.bets).index(np.max(self.bets))
        call_pot_size = self.pot_size - self.bets[fixed_hero_position] + np.max(self.bets)
        hero_range = prior_ranges[fixed_hero_position]
        villain_range = prior_ranges[villain_position]
        equity_bet_results = compute_weighted_equity_vector(hero_range, villain_range)

        # our equity of winning is all the p1 wins + half of the ties
        equity_hero_results = equity_bet_results[:, 0] + (0.5 * equity_bet_results[:, 2])

        # linear extrapolation
        lose_amount = -np.max(self.bets)
        win_amount = call_pot_size * (1 - self.rake) - np.max(self.bets)
        equity_amount = lose_amount + equity_hero_results * (win_amount - lose_amount)
        return equity_amount

    def _update_decision_chart(self, equity_matrix, step_size):
        optimal_decisions = np.argmax(equity_matrix, axis=0).ravel()

        optimal_decision_matrix = np.zeros((len(BetType), 169), dtype=np.float64)
        optimal_decision_matrix[optimal_decisions, list(range(169))] = 1

        # linearly extrapolate by step_size to make your iteration
        self.decision_chart = self.decision_chart * (1 - step_size) + optimal_decision_matrix * step_size

    def _update_decision_chart2(self, equity_matrix, step_size):
        valid_bets = sorted([bet.value for bet in self.valid_bets])
        median_equity = np.median(equity_matrix[valid_bets], axis=0).reshape((1, 169))
        min_equity = np.min(equity_matrix[valid_bets], axis=0).reshape((1, 169))
        # adjust probabilities based on equity diffs
        self.decision_chart[valid_bets] += step_size * (equity_matrix[valid_bets] - median_equity)
        self.decision_chart[valid_bets] = np.maximum(self.decision_chart[valid_bets], 0)
        self.decision_chart[valid_bets] /= np.sum(self.decision_chart[valid_bets], axis=0).reshape((1, 169))

        invalid_bets = sorted(set(range(len(BetType))).difference([bet.value for bet in self.valid_bets]))
        self.decision_chart[invalid_bets] = 0

    def equilibrium_step(self, step_size=0.1):
        """
        Make our way towards a Nash equilibrium.

        :param step_size: floating point number in the range of [0, 1] that represents
            how much we step towards the local optimum at each step.
        :return:
        """

        if self.maximum_vpip_players > 2:
            raise NotImplementedError("multi-way equity calculations not supported")

        if BetType.FOLD not in self.valid_bets:
            raise RuntimeError("folding should always be a valid bet type")

        # we're forced into a single decision, easy
        if len(self.valid_bets) < 2:
            only_valid_bet = self.valid_bets[0]
            self.decision_chart[:, :] = 0
            self.decision_chart[only_valid_bet.value] = 1
            return

        # go build the a-priori ranges based on all the previous decisions
        num_fixed_bets = len(self.bet_sequence)
        prior_ranges = np.ones((self.players, 169), dtype=np.float64)
        for bet in self.bet_sequence:
            step_range = bet.chart.decision_chart[bet.bet_type.value].ravel()
            prior_ranges[bet.chart.hero_position] *= step_range

        # go build an equity matrix of all the possible decisions you could do at this step
        equity_matrix = self._get_hypothetical_equity_matrix(
            num_fixed_bets=num_fixed_bets, fixed_hero_position=self.hero_position, prior_ranges=prior_ranges)

        # go figure out which decisions are the best
        invalid_bets = sorted(set(range(len(BetType))).difference([bet.value for bet in self.valid_bets]))
        equity_matrix[invalid_bets] = -np.inf

        # make your updates
        # self._update_decision_chart(equity_matrix, step_size)
        self._update_decision_chart2(equity_matrix, step_size)
