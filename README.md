### Solver to derive pre-flop decisions in poker

This is Michael Tong's attempt at building something that can quickly solve for GTO in poker.

### Main assumptions

* There is a CAP on the number of VPIP players for any given pot. The cap is currently 2.
  * (low priority) Hopefully the cap will be increased to 3 soon.
* Currently only `preflop_solver` is working. `post_flop_solver` is in development.
  * This means that in order to estimate the post-flop equity of any given position, I am using the raw equity numbers.
  * This is wrong because it doesn't take into account position, static/dynamic flops, stack size, or any other parameters associated with equity realization.
  * As of right now, if you run the pre-flop solver it undervalues suited cards and over-values A/KxO hands
* For pre-flop raises, there is only one possible raise size.
  * The raise size is currently raising to `2 * pot_size + 0.5BB`
  * (medium priority) I hope to support more bet sizes in the future
* When considering the head-to-equity values of 2 players, I ignore the ranges of all other players.
  * This is probably slightly wrong because the other players folding implies lower odds of having A/K/high-cards
  * (low priority) Figure out a good way to incorporate this into the solver. 

### How it works (high-level)

* For all combinations of head-to-head poker hole cards, I have cached the pre-flop equity results
  * The raw file can be seen in `resources/equity_raw_data.txt`
  * Instead of storing `ncr(52, 2) * ncr(50, 2)` combinations, I only store one combination per "suit rotation"
    * Suit rotations are taking advantage of the fact that 4 diamond cards are played the exact same way as 4 of any other suit
    * By doing this I can store ~16x less data.
    * The code to generate the equity calculations is in `equity_utils/equity_table.py`
* For pre-flop play, I construct the entire game tree of possible ways the pre-flop can go.
  * For 6 players and a max of 2 VPIP players, there are 110 possible places where a player can make a non-trivial decision
  * Each decision matrix is a `num_decisions x 169` matrix representing given for each combination of 169 cards, what is the probability of making each decision
  * The matrices are initialized to be uniform with each row summing to `1.0`
* To iterate on each solver towards a Nash equilibrium
  * For each (node, card combination)
  * Compute the estimated equity of making each decision
  * Compute a gradient associated with making the decision.
    * The gradient is `decision_equity - second_best_equity`
    * Add `gradient * step_size` to each decision probability, clip negatives to zero, re-normalize so the sum of probabilities is `1.0`
* Some other notes about what the solver
  * At each decision node there are `priors` vectors that take into account the probability of reaching a certain node.
    * For example, given the UTG raises, there is some chance all the downstream players may call/raise/fold
  * The solver takes into account the joint conditional probabilities associated with the hole cards of the fixed-player
    * For example, if we are considering what to do with `AA`, the probability that others have `Ax` is much lower.
  * In order to compute head-to-head equity, the solver is vectorized to take into account each combination of hole cards at the same time.
    * This is done by computing equity vectors to describe `estimated equity given each of the 169 possible hole card combinations`

### Future goals

From highest to lowest
* Figure out a way to model the post-flop play in an efficient manner.
* Figure out how to swap out the gradient-based solver to ADAM
* Incorporate more and varied betting strategies into the solver
* Support multi-way pot derivations.
* Figure out how to incorporate non-vpip decision-making into the equity calculation process
