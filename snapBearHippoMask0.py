## A team of N people enters a room and each is put on a BEAR mask or a HIPPO mask. The mask choices are randomly and independently determined. Each person is able to see other people's masks but cannot see his own mask. They cannot exchange any information with each other.
#
## Question: Could you help to calculate the following probability?
## (a) All N players are wearing the same types of masks
## (b) (N-1) players are wearing the same types of masks (while the rest player is wearing a different type)
## (c) you are also wearing a BEAR mask after observing all others are wearing BEAR masks
#########################################################################
#
## answers:
## (a) (1/2)^N + (1/2)^N
## (b) c(N, 1) * (1/2)^(N-1) * (1/2) * 2
## (c) 1/2

##-------------------------------------------

##The team now plays a game: these people are asked to guess their own masks simultaneously. Each of them can choose one of (1) guess BEAR (2) guess HIPPO (3) Don't make any guess. The team win the game if:
##- No one guesses incorrectly, AND
##- At least one person makes a guess and guesses correctly
##The team can meet before entering the room and agree on a strategy.
##
##Question-2: If we already have a strategy() implemented, could you write code to evaluate how good the strategy is?
##
##Assuming strategy() function is given:
##
##strategy(array<int> masks, N)
##arguments:
##   masks: masks[0], masks[1] ... is the mask put on to each player,
##   each mask can be 1 (BEAR) or -1 (Hippo)
##   N: number of players
## return:
##  a list of choices made by each player according to the strategy
##  each choice should be one of 1 (BEAR), -1 (Hippo), 0 (No Guess)
##
##To evaluate how good a given strategy is, we need to simulate the game under all possible mask configurations and calculate the probability that the team wins based on the strategy. The team wins if:
##
##No one guesses incorrectly, AND
##
##At least one person makes a correct guess.
##
##Approach
##1. Generate all possible mask configurations: For N players, each can be either BEAR (1) or HIPPO (-1), so there are
##2^N  possible configurations.
##
##2. Apply the strategy to each configuration: For each configuration, use the strategy() function to get the guesses of all players.
##
##3. Check the win condition: For each configuration, verify if the team wins based on the guesses.
##
##4. Calculate the winning probability: The probability is the number of winning configurations divided by the total number of configurations (
##2^N).
##
##Solution Code

import itertools


def evaluate_strategy(strategy, N):
    """
    Evaluates the effectiveness of a given strategy for N players.

    Args:
        strategy: A function that takes a list of masks and N, and returns a list of guesses.
        N: Number of players.

    Returns:
        float: The probability that the team wins under this strategy.
    """
    masks_values = [1, -1]  # 1 for BEAR, -1 for HIPPO
    total_configurations = 0
    winning_configurations = 0

    # Generate all possible mask configurations
    for masks in itertools.product(masks_values, repeat=N):
        total_configurations += 1
        guesses = strategy(list(masks), N)

        # Check if no one guesses incorrectly
        incorrect_guess = any(
            (guess != 0 and guess != masks[i])
            for i, guess in enumerate(guesses)
        )
        if incorrect_guess:
            continue

        # Check if at least one person makes a correct guess
        correct_guess = any(
            guess != 0 and guess == masks[i]
            for i, guess in enumerate(guesses)
        )
        if correct_guess:
            winning_configurations += 1

    # Calculate the winning probability
    winning_probability = winning_configurations / total_configurations
    return winning_probability

##Explanation
##1. Generating Configurations: The code uses itertools.product to generate all possible combinations of masks for N players. Each configuration is a tuple of 1 (BEAR) or -1 (HIPPO) of length N.
##2. Applying Strategy: For each configuration, the strategy() function is called to get the guesses of each player.
##3. Checking Win Conditions:
##    1) No Incorrect Guesses: The code checks if any player made a wrong guess (i.e., their guess is neither 0 nor their actual mask). If any wrong guess is found, the configuration is skipped.
##    2) At Least One Correct Guess: The code checks if at least one player made a correct guess (i.e., their guess matches their actual mask). If both conditions are met, the configuration is counted as a win.
##4. Calculating Probability: The winning probability is the ratio of winning configurations to the total number of configurations.
##This approach systematically evaluates the strategy across all possible scenarios, providing an exact measure of its effectiveness.

##----------------------------------------------------

##Optimal Strategy for N=3 Case
##For N=3 players, we can design a strategy that ensures the team wins in 6 out of 8 possible mask configurations, giving a win probability of 75%. Here's how it works:
##
##Strategy Rules
##Each player looks at the other two masks:
##
##If a player sees two identical masks (BEAR-BEAR or HIPPO-HIPPO), they guess the opposite of what they see.
##
##If a player sees two different masks (BEAR-HIPPO or HIPPO-BEAR), they do not guess (abstain).
##
##Why this works?
##
##If all three masks are the same (e.g., BEAR-BEAR-BEAR), all players will guess HIPPO (incorrectly), so the team loses.
##
##If two masks are the same and one is different, the player with the unique mask will correctly guess the opposite, while the other two abstain. The team wins.
##
##Example Scenarios
##Mask Configuration	Player 1 Sees	Player 2 Sees	Player 3 Sees	Guesses	Team Wins?
##BEAR, BEAR, BEAR	    B, B          	B, B	        B, B         	H, H, H	❌ (All wrong)
##BEAR, BEAR, HIPPO	    B, H          	B, H          	B, B         	0, 0, H	✅ (P3 correct)
##BEAR, HIPPO, BEAR	    H, B        	B, B          	B, H         	0, H, 0	✅ (P2 correct)
##HIPPO, BEAR, BEAR	    B, B        	H, B         	H, B        	H, 0, 0	✅ (P1 correct)
##HIPPO, HIPPO, HIPPO	H, H          	H, H         	H, H         	B, B, B	❌ (All wrong)
##HIPPO, HIPPO, BEAR	H, B          	H, B          	H, H        	0, 0, B	✅ (P3 correct)
##HIPPO, BEAR, HIPPO	B, H         	H, H         	H, B        	0, B, 0	✅ (P2 correct)
##BEAR, HIPPO, HIPPO	H, H        	B, H          	B, H        	B, 0, 0	✅ (P1 correct)
##Wins: 6/8 cases (✅)
##
##Losses: 2/8 cases (❌)
##
##Probability Calculation
##Total possible mask combinations:
##2^3=8
##Winning cases: 6
##Losing cases: 2 (all BEAR or all HIPPO)
##Winning probability:
##6 / 8 = 75%
##
##Why This is Optimal?
##No better deterministic strategy exists for N=3 because:
##If all players guess when they see two identical masks, they must lose when all masks are the same (since all guesses will be wrong).
##The only way to ensure at least one correct guess in mixed cases is to have exactly one player guess (the one with the unique mask).
##
##Alternative Strategies (For Comparison)
##All players guess randomly:
##Probability of winning: Very low (~12.5%).
##
##One fixed player always guesses BEAR, others abstain:
##Wins only if that player is BEAR (50% chance), but fails if others could have helped.
##
##This strategy strictly dominates alternatives by maximizing wins while minimizing losses.
##
##Conclusion
##For N=3, this strategy guarantees a 75% win rate, which is provably optimal. The key insight is:
##Activate guesses only when a player can logically deduce their mask (i.e., when the other two masks match).
##Avoid guessing when uncertain (i.e., when the other two masks differ).
##This balances risk and reward perfectly! 🎯


##Optimal Strategy for N=3 Players (Win Probability = 75%)
##Strategy Rules:
##Observation: Each player looks at the other two masks.
##
##Decision Rule:
##If a player sees two identical masks (BEAR-BEAR or HIPPO-HIPPO), they guess the opposite of what they see.
##If a player sees two different masks (BEAR-HIPPO or HIPPO-BEAR), they abstain (do not guess).
##
##Why This Works?
##We analyze all
##2^3=8 possible mask configurations and track the team's success/failure under this strategy.
##
##Mask Config (BEAR=B, HIPPO=H)            Player Actions                     Outcome
##BBB (All BEAR)                      All see BB → Guess H                Lose (All guess wrong)
##BBH                                 P1 (BB) → Guess H (correct)
##                                    P2 (BB) → Guess H (correct)
##                                    P3 (BH) → Abstain                   Win (2 correct guesses)
##BHB                                 P1 (BH) → Abstain
##                                    P2 (HB) → Abstain
##                                    P3 (BB) → Guess H (correct)         Win (1 correct guess)
##BHH                                 P1 (HH) → Guess B (correct)
##                                    P2 (BH) → Abstain
##                                    P3 (BH) → Abstain                   Win (1 correct guess)
##HBB                                 P1 (BB) → Guess H (correct)
##                                    P2 (HB) → Abstain
##                                    P3 (HB) → Abstain                   Win (1 correct guess)
##HBH                                 P1 (BH) → Abstain
##                                    P2 (HH) → Guess B (correct)
##                                    P3 (BH) → Abstain                   Win (1 correct guess)
##HHB                                    P1 (HB) → Abstain
##                                    P2 (HB) → Abstain
##                                    P3 (HH) → Guess B (correct)         Win (1 correct guess)
##HHH (All HIPPO)                     All see HH → Guess B                Lose (All guess wrong)
##Key Observations:
##Winning Cases (6/8):
##All non-uniform distributions (BBH, BHB, BHH, HBB, HBH, HHB) result in exactly one correct guess (others abstain).
##The team wins because only one player guesses correctly, and the rest abstain.
##
##Losing Cases (2/8):
##BBB or HHH: All players guess wrong (they see two identical masks and guess the opposite).
##
##Win Probability:
##p(Win)=6/8=75%
##
##Why This is Optimal:
##No strategy can achieve 100% because:
##If all players guess, they fail in uniform cases (BBB/HHH).
##If some abstain, they miss correct guesses in mixed cases.
##
##This strategy maximizes wins by ensuring:
##Correct guesses only in mixed cases (where the team needs them).
##No wasted guesses in uniform cases (where guesses would be wrong).
##Generalization to Larger N:
##
##For N players, similar strategies can be designed where:
##Players guess only if they see a specific pattern (e.g., N−1 identical masks).
##The win probability decreases as N grows but remains better than random guessing.
##
##Final Answer
##The strategy ensures:
##75% win rate for N=3.
##Losses only occur in BBB/HHH (all masks identical).
##Wins in all other cases (exactly one correct guess per winning configuration).
##This is provably optimal for N=3 under the given constraints. 🎯

