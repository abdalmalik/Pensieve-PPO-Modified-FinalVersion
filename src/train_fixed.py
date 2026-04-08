import multiprocessing as mp
import numpy as np
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from env import ABREnv
import ppo2 as network
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

S_DIM = [6, 8]
A_DIM = 6
ACTOR_LR_RATE = 1e-4
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 1000
TRAIN_EPOCH = 500000
MODEL_SAVE_INTERVAL = 300
RANDOM_SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_DIR = os.path.join(BASE_DIR, 'ppo')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
TRAIN_TRACES = os.path.join(BASE_DIR, 'train')
TEST_LOG_FOLDER = os.path.join(BASE_DIR, 'test_results')
LOG_FILE = SUMMARY_DIR + '/log'

if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

NN_MODEL = None    

def testing(epoch, nn_model, log_file):
    if os.path.exists(TEST_LOG_FOLDER):
        shutil.rmtree(TEST_LOG_FOLDER, ignore_errors=True)
    
    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    
    rewards, entropies = [], []
    if os.path.exists(TEST_LOG_FOLDER):
        test_log_files = os.listdir(TEST_LOG_FOLDER)
        for test_log_file in test_log_files:
            reward, entropy = [], []
            with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
                for line in f:
                    parse = line.split()
                    try:
                        entropy.append(float(parse[-2]))
                        reward.append(float(parse[-1]))
                    except IndexError:
                        break
            if reward:
                rewards.append(np.mean(reward[1:]))
            if entropy:
                entropies.append(np.mean(entropy[1:]))

    if rewards:
        rewards = np.array(rewards)
        rewards_min = np.min(rewards)
        rewards_5per = np.percentile(rewards, 5)
        rewards_mean = np.mean(rewards)
        rewards_median = np.percentile(rewards, 50)
        rewards_95per = np.percentile(rewards, 95)
        rewards_max = np.max(rewards)

        log_file.write(str(epoch) + '\t' +
                       str(rewards_min) + '\t' +
                       str(rewards_5per) + '\t' +
                       str(rewards_mean) + '\t' +
                       str(rewards_median) + '\t' +
                       str(rewards_95per) + '\t' +
                       str(rewards_max) + '\n')
        log_file.flush()
        return rewards_mean, np.mean(entropies) if entropies else 0
    return 0, 0
        
def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    
    with open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        actor = network.Network(state_dim=S_DIM, 
                                action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        writer = SummaryWriter(SUMMARY_DIR)

        nn_model = NN_MODEL
        if nn_model is not None:
            actor.load_model(nn_model)
            print('Model restored.')
        
        for epoch in range(TRAIN_EPOCH):
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s, a, p, r = [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a_, p_, r_ = exp_queues[i].get()
                s += s_
                a += a_
                p += p_
                r += r_
            s_batch = np.stack(s, axis=0)
            a_batch = np.vstack(a)
            p_batch = np.vstack(p)
            v_batch = np.vstack(r)

            actor.train(s_batch, a_batch, p_batch, v_batch, epoch)
            
            if epoch % MODEL_SAVE_INTERVAL == 0:
                actor.save_model(SUMMARY_DIR + '/nn_model_ep_' + str(epoch) + '.pth')
                print(f'Epoch {epoch}: Model saved')
                
                avg_reward, avg_entropy = testing(epoch,
                    SUMMARY_DIR + '/nn_model_ep_' + str(epoch) + '.pth', 
                    test_log_file)

                writer.add_scalar('Entropy Weight', actor._entropy_weight, epoch)
                writer.add_scalar('Reward', avg_reward, epoch)
                writer.add_scalar('Entropy', avg_entropy, epoch)
                writer.flush()

        final_model_path = SUMMARY_DIR + '/nn_model_ep_' + str(TRAIN_EPOCH) + '.pth'
        actor.save_model(final_model_path)
        print(f'Final model saved at: {final_model_path}')


def agent(agent_id, net_params_queue, exp_queue):
    env = ABREnv(agent_id)
    actor = network.Network(state_dim=S_DIM, action_dim=A_DIM,
                            learning_rate=ACTOR_LR_RATE)

    actor_net_params = net_params_queue.get()
    actor.set_network_params(actor_net_params)

    for epoch in range(TRAIN_EPOCH):
        obs = env.reset()
        s_batch, a_batch, p_batch, r_batch = [], [], [], []
        for step in range(TRAIN_SEQ_LEN):
            s_batch.append(obs)

            action_prob = actor.predict(
                np.reshape(obs, (1, S_DIM[0], S_DIM[1])))

            noise = np.random.gumbel(size=len(action_prob))
            bit_rate = np.argmax(np.log(action_prob) + noise)

            obs, rew, done, info = env.step(bit_rate)

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            a_batch.append(action_vec)
            r_batch.append(rew)
            p_batch.append(action_prob)
            if done:
                break
        v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
        exp_queue.put([s_batch, a_batch, p_batch, v_batch])

        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

def main():

    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(1)
    
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    coordinator.join()


if __name__ == '__main__':
    main()
