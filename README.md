# DDPG_TORCS2
DDPG with Gaussian noise for autonomous driving in TORCS


Driving a Car in TORCS using Deep Reinforcement Learning

We need noise at test time too for this to work; the training factors in the exploration noise, so noise has to be included at test time too; without it the car fails to complete a lap at test time

In the training phase, set irestart = 0 for restart from scratch;
If restarting from a ckpt file, then set irestart = 1 and restart_step = ckpt iteration number (e.g., 1300)


Youtube link:
https://www.youtube.com/watch?v=ajomz08hSIE

