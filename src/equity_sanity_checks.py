import time

import numpy as np

from src.card_utils.hole_cards import get_2d_index
from src.equity_utils.joint_equity_probabilities import compute_equity, \
    compute_weighted_equity, compute_weighted_equity_vector


def main():
    print('head to head', compute_equity())


    shit_cards = np.zeros(169, dtype=np.float64)
    # 27o
    shit_cards[get_2d_index(np.int8(0), np.int8(21))] = 0.5
    shit_cards[get_2d_index(np.int8(48), np.int8(49))] = 0.5

    good_cards = np.zeros(169, dtype=np.float64)
    # AA
    good_cards[get_2d_index(np.int8(48), np.int8(49))] = 1
    # AKS
    good_cards[get_2d_index(np.int8(44), np.int8(48))] = 1

    print(compute_weighted_equity_vector(good_cards, shit_cards)[: 5])
    print(compute_weighted_equity_vector(good_cards, shit_cards)[-5:])
    print('overall equity', compute_weighted_equity(good_cards, shit_cards))
    x = np.array([212248, 1493820, 6236])
    print([el / np.sum(x) for el in x])

    _ = compute_equity()

    t0 = time.time()
    for i in range(10000):
        _ = compute_equity()
    print(time.time() - t0)



if __name__ == '__main__':
    main()