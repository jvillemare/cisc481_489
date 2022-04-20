# Tileworld Research

Table of Contents:

 - [Initial Findings](#initial-findings)
 - [Questions](#questions)
 - [Runs](#runs)
 - [Theories](#theories)

## Initial Findings

 - The position of the wall and obstacles will always stay the same.
 - Each agent starts in the top left corner.
 - Position `[0, 0]` is the top-left corner.
 - Coins have values, 1 through 9.

The data we have access to looks like:

```commandline
>> print('Coin Data:', get_coin_data())
Coin Data: (
    [5, 2, 7, 2, 5, 8, 9, 9, 3, 2, 2], 
    [
        [100, 200], 
        [400, 400], 
        [500, 50], 
        [300, 350], 
        [50, 250], 
        [450, 400], 
        [450, 200], 
        [150, 0], 
        [150, 250], 
        [0, 100], 
        [150, 400]
    ]
)
>> print('Wall Positions:', get_wall_data())
Wall Positions: [
    [300, 50], 
    [200, 200], 
    [400, 200], 
    [300, 150], 
    [250, 400], 
    [350, 450], 
    [450, 100], 
    [350, 400], 
    [400, 450], 
    [100, 300], 
    [450, 250]
 ]
```

## Questions

 - Q: If both agents start in the top-left corner, does that count as a 
   collision, or is that ignored?
   - A: Not a collision
 - Q: Does the game world tick on regardless of how long the agents take, or 
   is it blocking? Meaning, how efficient does our computation have to be?
    - A: Ask the TA. Professor Decker **thinks** it blocks, but not sure.
 - Q: Does "wall" mean both the boundaries of the game board, or the 
   obstacles in the middle? Or both?
    - A: **Both**.
 - Q: Why is the coin and wall positions in the **hundreds?** 
    - A: The positions of the objects are **padded** by 50. So just 
      `position/50` and you'll get the true "normalized" position.
 - Q: When these assignments are being graded, then will the games be run 
   longer? 12 seconds is a short amount of time, and I think the randomness 
   will take over in that time.
    - A: Professor Decker is not sure, but it does not really matter since 
      the randomness is the same.
 - Q: I ran the `case = 1` (two random agents), and I got the exact same 
   score every time, what's up?
    - A: **The seed is not changing**. It's "random", but it is the same 
      random every time. Professor Decker says the TA will run with 
      different random seeds to compare against. The seed is set in `env.py`. 


## Runs

I ran the random agents (`case = 1`) a couple of times to calculate what the 
baseline is.

| # | Time | Size (`N`) | Seed | Agent1 Score | Agent2 Score | Total |
|---|------|------------|------|--------------|--------------|-------|
| 1 | 12   | 11         | 0    | 23           | 19           | 42    |
| 2 | 30   | 11         | 0    | -27          | -24          | -51   |
| 3 | 12   | 22         | 0    | 22           | 23           | 45    |
| 4 | 30   | 22         | 0    | 32           | 52           | 84    |


## Theories

 - **Opening idea:** Both agents will create a heat map of the board, where 
   coins (multiplied by their value) increase the "hotness" of an area, and 
   walls decrease the hotness of an area. Agent 1 goes for the center of the 
   hottest, and Agent 2 goes for the center of the second hottest area. 
   Their path to their designated hot areas can be augmented by "left" or 
   "right" by one if their is a coin of value 2 or greater.
