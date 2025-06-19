import time

import numpy as np

from src.gto_solver_old.chart import Chart
from src.gto_solver_old.constants import DEFAULT_BET_SIZES


def main():
    bet_sizes = DEFAULT_BET_SIZES

    ip_chart = Chart(3, 2, 0, bet_sizes, 0, 0)
    oop_chart = Chart(2, 3, 0, bet_sizes, 0, 0)

    ip_chart = Chart(3, 2, 1.5, bet_sizes, 0, 0)
    oop_chart = Chart(2, 3, 1.5, bet_sizes, 0, 0)


    t0 = time.time()

    for i in range(1000):
        ip_chart.optimize_chart(oop_chart, update_initial_bet=True, step_size=0.5)
        oop_chart.optimize_chart(ip_chart, update_initial_bet=True, step_size=0.5)

    print(time.time() - t0)
    print(ip_chart.hero_in_position)

    print(np.round([np.sum(ip_chart.chart[i]) for i in range(5)], 6))
    print(np.round([np.sum(oop_chart.chart[i]) for i in range(5)], 6))

    # ip_chart.optimize_chart(oop_chart, update_initial_bet=True, step_size=0.01, debug=True)


if __name__ == '__main__':
    main()