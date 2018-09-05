import logger
from Agent import Agent
from wrappers import make_env
import tensorflow as tf
import numpy as np
from collections import deque
from config import CONFIG as C
import pickle
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Only run on GPU 0

def reset_fs():
    fs = deque([], maxlen=C['frame_stack'])
    for _ in range(C['frame_stack']):
        fs.append(np.zeros((84,84), dtype=np.uint8))
    return fs

def main():
    logger.configure('./{}_logs'.format(C['env_id']))
    for k, v in C.items():
        logger.record_tabular(k, v)
    logger.dump_tabular()

    train_tracker = [0.0]
    eval_tracker = []
    best_reward = 0

    sess = tf.InteractiveSession()
    train_reward = tf.placeholder(tf.float32, name='train_reward')
    eval_reward = tf.placeholder(tf.float32, name='eval_reward')

    train_env = make_env(C['env_id'], C['noop_max'])
    eval_env = make_env(C['env_id'], C['noop_max'])
    agent = Agent(train_env, C)
    sess.run(tf.global_variables_initializer())
    agent.nn.update_target()

    train_summary = tf.summary.scalar('train_rew', train_reward)
    eval_summary = tf.summary.scalar('eval_reward', eval_reward)
    writer = tf.summary.FileWriter('{}{}_summary'.format('./', C['env_id']), sess.graph)

    train_fs = reset_fs()
    train_s = train_env.reset()
    for it in range(C['iterations']):
        # Training
        train_fs.append(train_s)
        train_a = agent.act(np.transpose(train_fs, (1,2,0)))
        ns, train_r, train_d, _ = train_env.step(train_a)
        train_tracker[-1] += train_r
        agent.perceive(train_s, train_a, train_r, float(train_d), it)
        train_s = ns
        if train_d:
            if train_env.env.env.was_real_done:
                if len(train_tracker) % 100 == 0:
                    summary = sess.run(train_summary, feed_dict={train_reward:np.mean(train_tracker[-100:])})
                    writer.add_summary(summary, it)
                    logger.record_tabular('steps', it)
                    logger.record_tabular('episode', len(train_tracker))
                    logger.record_tabular('epsilon', 100*agent.epsilon)
                    logger.record_tabular('learning rate', agent.lr)
                    logger.record_tabular('mean 100 episodes', np.mean(train_tracker[-100:]))
                    logger.dump_tabular()
                train_tracker.append(0.0)
            train_fs = reset_fs()
            train_s = train_env.reset()

        # Evaluation
        if it % C['eval_freq'] == 0:
            for _ in range(C['eval_episodes']):
                temp_video = []
                temp_reward = 0
                eval_tracker.append(0.0)
                eval_fs = reset_fs()
                eval_s = eval_env.reset()
                while True:
                    temp_video.append(eval_s)
                    eval_fs.append(eval_s)
                    eval_a = agent.greedy_act(np.transpose(eval_fs, (1,2,0)))
                    eval_s, eval_r, eval_d, _ = eval_env.step(eval_a)
                    eval_tracker[-1] += eval_r

                    if eval_env.env.env.was_real_done:
                        break
                    if eval_d:
                        eval_fs = reset_fs()
                        eval_s = eval_env.reset()

                if eval_tracker[-1] > best_reward: # Save best video
                    best_reward = eval_tracker[-1]
                    logger.log('Dump best video reward: {}'.format(best_reward))
                    best_video = temp_video
                    with open('video.pkl', 'wb') as f:
                        pickle.dump(best_video, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.log('Evaluate mean reward: {:.2f}, max reward: {:.2f}, std: {:.2f}'.format(np.mean(eval_tracker[-C['eval_episodes']:]), np.max(eval_tracker[-C['eval_episodes']:]), np.std(eval_tracker[-C['eval_episodes']:])))
            summary = sess.run(eval_summary, feed_dict={eval_reward:np.mean(eval_tracker[-C['eval_episodes']:])})
            writer.add_summary(summary, it)

    agent.nn.save('./{}_model'.format(C['env_id']))

if __name__ == '__main__':
    main()
