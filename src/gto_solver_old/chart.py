from typing import Optional

import numpy as np

from src.equity_utils.joint_equity_probabilities import compute_weighted_equity_vector, \
    load_conditional_joint_probabilities
from src.gto_solver_old.constants import check_hero_wins


# TODO: figure out a way to derive the SB/BB charts
class Chart:
    """
    Prototype of chart class. This was designed for head to head play and really doesn't scale...
    Need to redesign the whole thing
    """
    def __init__(self,
                 hero_position: int, villain_position: int, extra_in_pot: float,
                 bet_sizes: list[float], hero_non_vpip: float, villain_non_vpip: float):
        """
        positions start at 0 for first to act, then increment up to num_players-1 for the BB
        Money is in units of BB.

        Assumes that:
            * No tricky/slow play in preflop
            * The only decisions will be raise/call/fold of pre-set bet sizes

        :param hero_position: position of the hero
        :param villain_position: position of the villain
        :param extra_in_pot: extra amount of money already in pot not belonging to the hero or villain
        :param bet_sizes: total bet sizes. will be in units of money put in that includes money already put in pot
            So for example a 3 bet includes the amount already in the initial raise.
            Can also be thought of as `how much money do you put in for pre-flop`
        :param hero_non_vpip: amount of money that hero did not voluntarily put in pot
        :param villain_non_vpip: amount of money that hero did not voluntarily put in pot
        """
        self.hero_position = hero_position
        self.villain_position = villain_position
        self.extra_in_pot = extra_in_pot
        self.bet_sizes = bet_sizes
        self.bet_sizes = bet_sizes
        self.hero_non_vpip = hero_non_vpip
        self.villain_non_vpip = villain_non_vpip

        # whether the hero is the last to act
        self.hero_in_position = self.hero_position > self.villain_position

        # if the fold decision is not included, add a fold bet
        if self.bet_sizes[0] != self.hero_non_vpip:
            self.bet_sizes = [self.hero_non_vpip] + self.bet_sizes

        if len(self.bet_sizes) > 5:
            raise NotImplementedError("6+ betting not supported, need to update gto_solver_old.constants.BET_RESOLUTIONS")

        # the master chart of what to do, where each a value of (i,j,k) is given
        # cards (j, k) you'd be willing to bet up to the i'th round a certain percentage of times.
        # things like (1, j, k) out of position means you'd call against a raise
        # (1, j, k) in position means you'd rfi
        self.chart = np.ones((len(self.bet_sizes), 13, 13), dtype=np.float64)

        # initialize the chart as being willing to be equally likely to bet to each round in each position
        self.chart = self.chart / len(self.bet_sizes)

    def _compute_equity_results(self, other: 'Chart'):
        """
        compute the equity against the opponent's strategy for each set of cards. Indexes correspond to
        For index [i, j, k]
        If you have card combination (j, k) and you are in betting round i against the villain, what is your equity


        :param other: the other chart
        """
        # compute the equity against the opponent's strategy for each set of cards
        equity_results = np.zeros((len(self.bet_sizes), 13, 13), dtype=np.float64)
        for bet_index in range(len(self.bet_sizes)):
            # compute the equity of calling in each position against the opponent's range
            calling_range = np.ones(169, dtype=np.float64)
            other_range = other.chart[bet_index].ravel()

            equity_bet_results = compute_weighted_equity_vector(calling_range, other_range)
            # our equity of winning is all the p1 wins + half of the ties
            equity_hero_results = equity_bet_results[:, 0] + (0.5 * equity_bet_results[:, 2])
            equity_hero_results = equity_hero_results.reshape((13, 13))
            equity_results[bet_index] = equity_hero_results
        return equity_results

    def _compute_equity_matrices(self):
        """
        Compute some basic equity assuming either the hero/villain folds.

        If the hero is willing to bet index i and the villain bets index j, then...

        Folding equity matrix: if non-zero here's how much the hero wins/loses (negative)
        needs_equity_matrix: neither player folds, we raise a flag to do the equity calculations.
        """
        # compute the equity assuming the hero/villain folds
        folding_equity_matrix = np.zeros((len(self.bet_sizes), len(self.bet_sizes)), dtype=np.float64)
        needs_equity_matrix = np.zeros((len(self.bet_sizes), len(self.bet_sizes)), dtype=np.float64)
        for hero_bet_index in range(len(self.bet_sizes)):
            for villain_bet_index in range(len(self.bet_sizes)):
                hero_wins = check_hero_wins(hero_bet_index, villain_bet_index, self.hero_in_position)
                villain_wins = check_hero_wins(villain_bet_index, hero_bet_index, not self.hero_in_position)

                if not hero_wins and not villain_wins:
                    needs_equity_matrix[hero_bet_index, villain_bet_index] = 1.0

                bet_amount = self.bet_sizes[min(hero_bet_index, villain_bet_index)]
                if hero_wins:
                    folding_equity_matrix[hero_bet_index, villain_bet_index] = self.extra_in_pot + bet_amount
                elif villain_wins:
                    folding_equity_matrix[hero_bet_index, villain_bet_index] = - bet_amount
        return folding_equity_matrix, needs_equity_matrix

    def _compute_conditional_bet_chances(self):
        """
        compute the conditional bet chances the opponent's strategy for each set of cards. Indexes correspond to
        For index [i, j, k]
        If you have card combination (j, k) what is the chance of the villain betting i
        """
        conditional_joint_probabilities = load_conditional_joint_probabilities()
        conditional_bet_chances = np.zeros((len(self.bet_sizes), 13, 13), dtype=np.float64)

        # for each possible bet
        for bet_index in range(len(self.bet_sizes)):
            # take the current bet probabilities
            flattened_bet_chart = self.chart[bet_index].reshape((-1, 1))
            # multiply by the joint conditional probabilities. Use matrix math for speed
            bet_conditional_chances = np.dot(conditional_joint_probabilities, flattened_bet_chart)
            # reshape and assign
            bet_conditional_chances = bet_conditional_chances.reshape(13, 13)
            conditional_bet_chances[bet_index] = bet_conditional_chances

        return conditional_bet_chances

    def _compute_ev_results(self, conditional_bet_chances, equity_results, folding_equity_matrix, needs_equity_matrix, debug=False):
        """
        Compute the final EV matrix of all the decisions. For a result of [i, j, k, l] the results are:

        For a hero willing to bet i,
            a villain willing to bet j,
            and hero cards being (k, l)

        What is the expected value coming from this decision.

        Note here is where I also include the joint probability of the villain being willing to make bet j
        given the hero cards (k, l)
        """
        # compute the EV of each decision, matrix dimensions are:
        # hero_bet_index, villain_bet_index, 169 card positions for hero
        ev_results = np.zeros((len(self.bet_sizes), len(self.bet_sizes), 13, 13), dtype=np.float64)
        for hero_bet_index in range(len(self.bet_sizes)):
            for villain_bet_index in range(len(self.bet_sizes)):
                needs_equity = needs_equity_matrix[hero_bet_index, villain_bet_index]

                # no need to compute equity, just plug in a constant for the value of the hand given folding
                if not needs_equity:
                    ev_results[hero_bet_index, villain_bet_index] = folding_equity_matrix[
                        hero_bet_index, villain_bet_index]
                    continue

                # amount of money won and lost if at zero and full (100%) equity respectively
                bet_index = min(hero_bet_index, villain_bet_index)
                zero_equity = -self.bet_sizes[bet_index]
                full_equity = self.bet_sizes[bet_index] * 2 + self.extra_in_pot

                # linear interpolation between the two points
                ev_results[hero_bet_index, villain_bet_index] = (
                        zero_equity +
                        (full_equity * equity_results[hero_bet_index])
                )
        # need to multiply the EV matrix by conditionals that villain will be in each of these positions
        for villain_bet_index in range(len(self.bet_sizes)):
            ev_results[:, villain_bet_index] *= conditional_bet_chances[villain_bet_index]

        if debug:
            debug_index = [0, 6]
            assert False, (
                folding_equity_matrix, needs_equity_matrix,
                ev_results.shape,
                conditional_bet_chances[:, debug_index[0], debug_index[1]],
                np.round(ev_results[:, :, debug_index[0], debug_index[1]], 5),
                np.round(ev_results.sum(axis=1)[:, debug_index[0], debug_index[1]], 5)
            )

        # sum the ev_results to get the ev for each of the hero choices summed over the villain choices
        ev_results = ev_results.sum(axis=1)
        return ev_results

    def _compute_decision_matrix(self, ev_results, update_initial_bet):
        """
        Now that we have the EV for each decision, pick the decision that gives you the maximum EV
        """
        new_decision_matrix = np.zeros(self.chart.shape, dtype=self.chart.dtype)
        if update_initial_bet:
            best_decision = np.argmax(ev_results, axis=0)
            for i in range(13):
                for j in range(13):
                    new_decision_matrix[best_decision[i, j], i, j] = 1.0
        # keep the initial bet fixed, update everything else
        else:
            best_decision = np.argmax(ev_results[1:], axis=0)
            new_decision_matrix[0] = self.chart[0]

            for i in range(13):
                for j in range(13):
                    new_decision_matrix[best_decision[i, j] + 1, i, j] = 1.0 - self.chart[0, i, j]
        return new_decision_matrix

    def optimize_chart(self, other: 'Chart', update_initial_bet=False, step_size=0.5, debug=False) -> None:
        """
        Optimize the preflop decision chart.

        :param other: the other chart to optimize against
        :param update_initial_bet: whether to update initial round of betting. In other words,
            whether to update chart[:, :, 0]
        :param step_size: given the local optimal point and the current point how much to step to the
            new step size. Formula is `new <- old (1 - step_size) + new (step_size)`
        """
        if not isinstance(other, Chart):
            raise RuntimeError("Other sould be a chart instance")

        if self.bet_sizes[1:] != other.bet_sizes[1:]:
            raise RuntimeError("All post-initial bets should match")

        if len(self.bet_sizes) != len(other.bet_sizes):
            raise RuntimeError("bet_sizes lengths should match")

        if self.hero_position == other.hero_position:
            raise RuntimeError("hero_position values should not match")

        # compute the equity against the opponent's strategy for each set of cards
        equity_results = self._compute_equity_results(other)

        # compute the equity assuming the hero/villain folds
        folding_equity_matrix, needs_equity_matrix = self._compute_equity_matrices()

        # figure out the conditional probabilities of the villain betting i given cards (j, k)
        conditional_bet_chances = self._compute_conditional_bet_chances()

        # compute the EV of each decision, matrix dimensions are:
        # hero_bet_index, villain_bet_index, 13x13 card positions for hero
        ev_results = self._compute_ev_results(
            conditional_bet_chances, equity_results, folding_equity_matrix, needs_equity_matrix, debug)

        # Figure out what are the best decisions to make at each step
        new_decision_matrix = self._compute_decision_matrix(ev_results, update_initial_bet)

        # finally, perform interpolation
        self.chart = step_size * new_decision_matrix + (1 - step_size) * self.chart

        if debug:
            debug_index = [0, 6]
            assert False, (
                folding_equity_matrix, needs_equity_matrix,
                ev_results[:, debug_index[0], debug_index[1]],
                new_decision_matrix[:, debug_index[0], debug_index[1]],
            )
