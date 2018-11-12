# Deep Q Network
<img src="video.gif" width="160" height="210" />
Reproduce performance (rewards) of the following deep reinforcement learning methods using TensorFlow:

+ Deep Q Network (DQN):
[Human-level Control Through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

+ Dueling DQN:
[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)

+ Double DQN:
[Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/pdf/1509.06461.pdf)

This is an easy to understand and modify the DQN structure as well as memory efficiency implementation that can store 1M transitions using ~8GB memory. The model architecture used in this code is not quite similar to the one described in the original DQN paper. I have tested with Pong, Breakout, and MsPacman so far.

It took ~25 hours of training to reach its first 400 points reward on Breakout evaluation using 1 GTX 1080.

## Environment I used
- Python 3.6
- TensorFlow 1.10
- OpenAI Gym 0.10.5
- OpenCV 3.4.2
- mpi4py 3.0.0

## How to run
Set up hyper-parameters in [config.py](./config.py). To run the program:
```
python train.py
```
