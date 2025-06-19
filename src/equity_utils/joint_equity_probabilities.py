"""
Compute some matrices to determine:
* Given you 13^2 possible starting cards, what is the probability the opponent has those cards.
* For each of those 13^4 combinations, figure out what is the equity of the position.
"""
import functools
import os
from pathlib import Path

import numpy as np
from numba import jit

from src.card_utils import hole_cards
from src.equity_utils.equity_table import load_head_to_head_equity

HEAD_TO_HEAD_JOINT_PROBABILITY_PATH = 'head_to_head_joint_probability.npy'
HEAD_TO_HEAD_WINS_PROBABILITY_PATH = 'head_to_head_wins_probability.npy'

# these should be loaded from main, cast things to float64 to do fancy vector stuff...
@functools.cache
def load_head_to_head_joint_probability():
    """
    Returns a matrix representing given the hole_cards at i and hole_cards at j
    how many possible combinations of cards exist
    """
    src_path = Path(os.path.dirname(__file__)).parent
    joint_probability = np.load(f'{src_path}/resources/{HEAD_TO_HEAD_JOINT_PROBABILITY_PATH}')
    return np.array(joint_probability, dtype=np.float64)

@functools.cache
def load_conditional_joint_probabilities():
    """
    Return a 169x169 matrix of given each hole_cards set at i, what is the joint conditional
    probability that there will be the other j combinations of cards
    """
    joint_probability = load_head_to_head_joint_probability()
    return joint_probability / np.sum(joint_probability, axis=1).reshape((-1, 1))

@functools.cache
def load_head_to_head_wins_probability():
    """
    Given hole cards at i and other hole cards at j return a (i wins, j wins, they tie)
    probability vector.
    """
    src_path = Path(os.path.dirname(__file__)).parent
    wins_probability = np.load(f'{src_path}/resources/{HEAD_TO_HEAD_WINS_PROBABILITY_PATH}')
    return np.array(wins_probability, dtype=np.float64)

def compute_weighted_equity_matrices(p1_vector, p2_vector):
    """
    Compute the equity for two weighted probabilities.
    This is kind of slow (10k runs/second) so maybe see if we should speed this up later

    :param p1_vector: 169 vector of probabilities that a player will be here
    :param p2_vector: 169 vector of probabilities that a player will be here
    :return: p1/p2/tie equity vector
    """
    # compute the joint probabilities without having to explicitly create the outer product
    weighted_joint = (
        load_head_to_head_joint_probability()
        * p1_vector.reshape((-1, 1))
        * p2_vector.reshape((1, -1))
    )

    weighted_wins = load_head_to_head_wins_probability().copy()

    for i in range(3):
        weighted_wins[:, :, i] *= weighted_joint

    return weighted_joint, weighted_wins

def compute_weighted_equity_vector(p1_vector, p2_vector):
    """
    Given some p1/p2 ranges, compute the equity of p1/p2/tie happening for each possible card in p1's range

    :param p1_vector: 169 vector of a range
    :param p2_vector: 169 vector of a range
    :return: 3x169 vector of the chances of p1/p2/tie with equity ranges
    """
    weighted_joint, weighted_wins = compute_weighted_equity_matrices(p1_vector, p2_vector)
    return jit_compute_weighted_equity_vector(weighted_joint, weighted_wins)

@jit(nopython=True)
def jit_compute_weighted_equity_vector(weighted_joint, weighted_wins):
    """
    jit helper function to compute_weighted_equity_vector
        """
    weighted_wins_vector = weighted_wins.sum(axis=1)
    non_zero_mask = weighted_wins_vector.sum(axis=1) > 0
    joint_vector = np.sum(weighted_joint, axis=1).reshape((-1, 1))
    weighted_wins_vector[non_zero_mask] = weighted_wins_vector[non_zero_mask] / joint_vector[non_zero_mask]
    return weighted_wins_vector

def compute_weighted_equity(p1_vector, p2_vector):
    weighted_joint, weighted_wins = compute_weighted_equity_matrices(p1_vector, p2_vector)
    weighted_equity = weighted_wins.sum(axis=1).sum(axis=0) / np.sum(weighted_joint)
    return weighted_equity

def compute_equity():
    """
    Mostly for debugging. Compute the equity of two players just blindly calling with everything
    """
    ones_vector = np.ones(169, dtype=np.float64)
    return compute_weighted_equity(ones_vector, ones_vector)

def main():
    """
    Given all the suit rotations, distill the information into the (13^2)^2 combinations that actually
    affect play
    """
    equity_data = load_head_to_head_equity()

    head_to_head_joint_probability = np.zeros(shape=(169, 169), dtype=np.int32)
    head_to_head_wins_probability = np.zeros(shape=(169, 169, 3), dtype=np.float64)

    for suit_rotation, num_combos, equity in equity_data.values:
        hole_rank = hole_cards.get_2d_index(np.int8(suit_rotation[0]), np.int8(suit_rotation[1]))
        other_hole_rank = hole_cards.get_2d_index(np.int8(suit_rotation[2]), np.int8(suit_rotation[3]))

        head_to_head_joint_probability[hole_rank, other_hole_rank] += num_combos
        # count the total number of times you win/lose/tie
        head_to_head_wins_probability[hole_rank, other_hole_rank] += equity * num_combos

    # normalize all the wins probabilities to sum to one
    for i in range(169):
        for j in range(169):
            equity = head_to_head_wins_probability[i, j]
            head_to_head_wins_probability[i, j] = equity / np.sum(equity)

    np.save(f'../resources/{HEAD_TO_HEAD_JOINT_PROBABILITY_PATH}', head_to_head_joint_probability)
    np.save(f'../resources/{HEAD_TO_HEAD_WINS_PROBABILITY_PATH}', head_to_head_wins_probability)

if __name__ == '__main__':
    main()