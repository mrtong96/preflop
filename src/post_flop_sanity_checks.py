import numpy as np

from src.post_flop_solver.postflop_chart import PostflopChart

if __name__ == '__main__':
    def count_children(node: PostflopChart):
        return 1 + sum([count_children(child_node) for child_node in node.children_charts])

    base_chart = PostflopChart(
        community_cards=np.array([0, 1, 2]),
        bet_sizes=np.array([1, 1, 0]),
        stack_size=100,
        ip_hole_card_priors = np.ones(169),
        oop_hole_card_priors=np.ones(169),
        ip_to_act=False,
        is_first_street_action=True,
    )

    print(count_children(base_chart))

    print('hi')