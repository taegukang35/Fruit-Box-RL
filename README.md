# Fruit-Box-RL

## DONE
* Implemented direct value function; learning becomes consistently increasing
* Fixed bug on env; computing legal actions seem to make it not work with rllib

## WIP
* Waiting for full results for modified value function + CNN

## TODO
* Implement policy subclass --- Code is wrong, no way to discourage illegal actions due to the representation of action space + we cant know the action is illegal or not before actually deciding one of the points
* Try out negative reward with direct value function

### How can we discourage illegal actions?
1. Give negative rewards for illegal actions
2. No negative rewards, but include in the loss function instead -> need to modify PPO algorithm
3. Modify the env s.t. currently illegal actions are clipped to legal boundaries instead - Dont know if this actually reduces search space
