# who cares if it's slow
import itertools
from collections import Counter
from typing import Optional

import numpy as np

from src.card_utils.card import get_rank, get_suit


def get_suit_rotation(cards: tuple[int, ...]) -> tuple[str, ...]:
    cards = np.array(cards, dtype=np.int8)
    ranks = [get_rank(card) for card in cards]
    suits = []
    suit_mapping = dict()
    for card in cards:
        if get_suit(card) not in suit_mapping:
            suit_mapping[get_suit(card)] = len(suit_mapping)
        suits.append(suit_mapping[get_suit(card)])

    cards = [rank << 2 | suit for rank, suit in zip(ranks, suits)]
    return tuple(cards)

def get_suit_rotation2(community_cards: tuple[int, ...], hole_cards: Optional[tuple[int, ...]]=None) -> tuple[int, ...]:
    suit_min = None
    if hole_cards is None:
        if len(community_cards) == 3:
            suit_min = 1
        elif len(community_cards) == 4:
            suit_min = 2
        elif len(community_cards) == 5:
            suit_min = 3
    else:
        if len(community_cards) == 3:
            suit_min = 3
        elif len(community_cards) == 4:
            suit_min = 4
        elif len(community_cards) == 5:
            suit_min = 5

    assert suit_min is not None

    combined_cards = list(community_cards)
    if hole_cards is not None:
        combined_cards += list(hole_cards)
    combined_cards = np.array(combined_cards, dtype=np.int8)
    ranks = [get_rank(card) for card in combined_cards]
    suits = [get_suit(card) for card in combined_cards]

    suit_counts = Counter(suits)
    suit_mapping = dict()
    suit_index = 0
    for suit, count in suit_counts.items():
        if count >= suit_min:
            suit_mapping[suit] = suit_index
            suit_index += 1

    cards = [rank << 2 | suit_mapping.get(suit, 3) for rank, suit in zip(ranks, suits)]
    return tuple(cards)

def main():

    for community_cards in [3, 4, 5]:
        print('-' * 80)

        print('community cards', community_cards)
        suit_rotation_count = Counter()

        deck = list(range(52))
        for cards in itertools.combinations(deck, community_cards):
            suit_rotation = get_suit_rotation2(cards)
            suit_rotation_count[suit_rotation] += 1

        print(len(suit_rotation_count), sum(suit_rotation_count.values()), sum(suit_rotation_count.values()) / len(suit_rotation_count))

        total_count = 0
        for rotation in suit_rotation_count.keys():
            hole_card_set = set()
            remaining_cards = [card for card in range(52) if card not in rotation]
            for hole_cards in itertools.combinations(remaining_cards, 2):
                hole_card_set.add(get_suit_rotation2(rotation, hole_cards))

            total_count += len(hole_card_set)

        print(len(suit_rotation_count), total_count, total_count / len(suit_rotation_count))

if __name__ == '__main__':
    main()

'''
--------------------------------------------------------------------------------
community cards 3
1911 22100 11.564625850340136
1911 1477034 772.9115646258504
--------------------------------------------------------------------------------
community cards 4
20410 270725 13.264331210191083
20410 18682560 915.3630573248407
--------------------------------------------------------------------------------
community cards 5
204087 2598960 12.734569080833174
204087 200780658 983.7993502770877
--------------------------------------------------------------------------------
community cards 3
1911 22100 11.564625850340136
1911 626626 327.9047619047619
--------------------------------------------------------------------------------
community cards 4
20410 270725 13.264331210191083
20410 4553744 223.11337579617833
--------------------------------------------------------------------------------
community cards 5
58513 2598960 44.41679626749611
58513 10074142 172.16929571206398
'''