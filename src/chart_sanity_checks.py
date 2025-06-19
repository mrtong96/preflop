import time

import numpy as np
from PIL.ImageOps import scale

from src.preflop_solver.chart import Chart

np.set_printoptions(linewidth=180)

NUM_PLAYERS = 6

def count_children(chart):
    # print(len(chart.children_charts), chart.bet_sequence)
    return 1 + sum([count_children(child) for child in chart.children_charts])

def run_equilibrium_step(chart, step_size=0.01):
    chart.equilibrium_step(step_size)
    for child_chart in chart.children_charts:
        run_equilibrium_step(child_chart, step_size)

def main():
    bets = np.zeros((NUM_PLAYERS,), dtype=np.float64)
    bets[-2] = 0.5
    bets[-1] = 1.0
    is_in = np.ones((NUM_PLAYERS,), dtype=np.bool_)
    vpip = np.zeros((NUM_PLAYERS,), dtype=np.bool_)

    root_chart = Chart(
        hero_position=0,
        bets=bets,
        is_in=is_in,
        vpip=vpip,
        max_vpip_players=2,
        players=NUM_PLAYERS,
        stack_size=100,
        rake=0.00,
    )
    print(count_children(root_chart))

    t0 = time.time()
    step_size = 0.1
    scale_factor = 1
    for i in range(2000):
        run_equilibrium_step(root_chart, step_size=step_size)
        if i % 100 == 0:
            decision_chart = root_chart.decision_chart.copy()
        elif i % 100 == 1:
            decision_chart2 = root_chart.decision_chart.copy()
            diff = (decision_chart2 - decision_chart) * scale_factor
            print(i, time.time() - t0, 'diff', np.mean(np.sqrt(np.sum(np.square(diff)))))

        if i != 0 and i % 200 == 0:
            step_size /= 2
            scale_factor *= 2
    print('total time', time.time() - t0)

    # print(np.round(root_chart.decision_chart, 3).reshape(3, 13, 13))
    print(np.round(root_chart.decision_chart, 3)[2].reshape(13, 13)[::-1, ::-1])

    totals = []
    for i in range(3):
        total = 0
        decision = root_chart.decision_chart[i].reshape(13, 13)[::-1, ::-1]
        for j in range(13):
            for k in range(13):
                if j < k:
                    total += decision[j, k] * 4
                elif j == k:
                    total += decision[j, k] * 6
                else:
                    total += decision[j, k] * 12
        totals.append(total)
    print(totals)

if __name__ == '__main__':
    main()