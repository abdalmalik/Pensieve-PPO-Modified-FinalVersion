import multiprocessing as mp
import numpy as np
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from env import ABREnv
import ppo2 as network
import torch

# ============================================================================
# SECTION 1: ENVIRONMENT CONFIGURATION
# ============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ============================================================================
# SECTION 2: HYPERPARAMETERS
# ============================================================================
S_DIM = [6, 8]                      # State dimension
A_DIM = 6                           # Action dimension (bitrates)
ACTOR_LR_RATE = 1e-4                # Learning rate
NUM_AGENTS = 16                     # Number of parallel agents
TRAIN_SEQ_LEN = 1000                # Training sequence length
TRAIN_EPOCH = 500000                # Total training epochs
MODEL_SAVE_INTERVAL = 300           # Save model every N epochs
RANDOM_SEED = 42                    # Random seed for reproducibility

# ============================================================================
# SECTION 3: PATH CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_DIR = os.path.join(BASE_DIR, 'ppo')               # Directory for checkpoints and logs
MODEL_DIR = os.path.join(BASE_DIR, 'models')              # Directory for final models
TRAIN_TRACES = os.path.join(BASE_DIR, 'train')            # Training traces directory
TEST_LOG_FOLDER = os.path.join(BASE_DIR, 'test_results')  # Test results directory
LOG_FILE = SUMMARY_DIR + '/log'     # Log file path

# ============================================================================
# SECTION 4: LOAD LAST CHECKPOINT
# ============================================================================
LAST_MODEL_EPOCH = 500000            # Last saved model epoch (CHANGED FROM 17700 TO 99900)
NN_MODEL = f'{SUMMARY_DIR}/nn_model_ep_{LAST_MODEL_EPOCH}.pth'

# Create directories if they don't exist
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

# ============================================================================
# SECTION 5: TESTING FUNCTION
# ============================================================================
def testing(epoch, nn_model, log_file):
    """
    Evaluate model performance on test traces
    """
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

# ============================================================================
# SECTION 6: CENTRAL AGENT (TRAINING COORDINATOR)
# ============================================================================
def central_agent(net_params_queues, exp_queues):
    """
    Central agent that manages training across all workers
    """
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    
    with open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        # Initialize actor network
        actor = network.Network(
            state_dim=S_DIM, 
            action_dim=A_DIM,
            learning_rate=ACTOR_LR_RATE
        )

        writer = SummaryWriter(SUMMARY_DIR)

        # --------------------------------------------------------------------
        # Load saved model if exists
        # --------------------------------------------------------------------
        nn_model = NN_MODEL
        start_epoch = 0
        
        if nn_model is not None and os.path.exists(nn_model):
            actor.load_model(nn_model)
            start_epoch = LAST_MODEL_EPOCH + 1
            print(f'Model loaded from: {nn_model}')
            print(f'Resuming training from epoch: {start_epoch}')
            print(f'Remaining epochs: {TRAIN_EPOCH - start_epoch}')
        else:
            print('Model not found! Starting training from scratch.')
            start_epoch = 0
        
        # --------------------------------------------------------------------
        # Main training loop
        # --------------------------------------------------------------------
        for epoch in range(start_epoch, TRAIN_EPOCH):
            # Send current network parameters to all agents
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            # Collect experiences from all agents
            s, a, p, r = [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a_, p_, r_ = exp_queues[i].get()
                s += s_
                a += a_
                p += p_
                r += r_
            
            # Prepare batches for training
            s_batch = np.stack(s, axis=0)
            a_batch = np.vstack(a)
            p_batch = np.vstack(p)
            v_batch = np.vstack(r)

            # Train the actor network
            actor.train(s_batch, a_batch, p_batch, v_batch, epoch)
            
            # --------------------------------------------------------------------
            # Save checkpoint and run evaluation
            # --------------------------------------------------------------------
            if epoch % MODEL_SAVE_INTERVAL == 0:
                actor.save_model(f'{SUMMARY_DIR}/nn_model_ep_{epoch}.pth')
                print(f'Epoch {epoch}: Model saved')
                
                avg_reward, avg_entropy = testing(
                    epoch,
                    f'{SUMMARY_DIR}/nn_model_ep_{epoch}.pth', 
                    test_log_file
                )

                writer.add_scalar('Entropy Weight', actor._entropy_weight, epoch)
                writer.add_scalar('Reward', avg_reward, epoch)
                writer.add_scalar('Entropy', avg_entropy, epoch)
                writer.flush()
                
                print(f'   - Average reward: {avg_reward:.2f}')
                print(f'   - Entropy: {avg_entropy:.4f}')
            
            # Progress indicator
            if epoch % 100 == 0:
                print(f'Training in progress... Epoch: {epoch}/{TRAIN_EPOCH}')

        final_model_path = f'{SUMMARY_DIR}/nn_model_ep_{TRAIN_EPOCH}.pth'
        actor.save_model(final_model_path)
        print(f'Final model saved at: {final_model_path}')

# ============================================================================
# SECTION 7: LOCAL AGENT (ENVIRONMENT INTERACTION)
# ============================================================================
def agent(agent_id, net_params_queue, exp_queue, start_epoch):
    """
    Local agent that interacts with environment and collects experiences
    """
    env = ABREnv(agent_id)
    actor = network.Network(
        state_dim=S_DIM, 
        action_dim=A_DIM,
        learning_rate=ACTOR_LR_RATE
    )

    # Main agent loop - START FROM start_epoch (FIXED)
    for epoch in range(start_epoch, TRAIN_EPOCH):
        # Get latest network parameters
        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        
        # Reset environment
        obs = env.reset()
        s_batch, a_batch, p_batch, r_batch = [], [], [], []
        
        # Collect trajectory
        for step in range(TRAIN_SEQ_LEN):
            s_batch.append(obs)

            # Sample action using Gumbel-Softmax
            action_prob = actor.predict(
                np.reshape(obs, (1, S_DIM[0], S_DIM[1]))
            )
            noise = np.random.gumbel(size=len(action_prob))
            bit_rate = np.argmax(np.log(action_prob) + noise)

            # Take step in environment
            obs, rew, done, info = env.step(bit_rate)

            # Store experience
            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            a_batch.append(action_vec)
            r_batch.append(rew)
            p_batch.append(action_prob)
            
            if done:
                break
        
        # Compute values and send to central agent
        v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
        exp_queue.put([s_batch, a_batch, p_batch, v_batch])

# ============================================================================
# SECTION 8: MAIN FUNCTION
# ============================================================================
def main():
    """
    Main entry point - sets up multiprocessing and starts training
    """
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(1)
    
    # Determine starting epoch
    start_epoch = LAST_MODEL_EPOCH + 1 if os.path.exists(NN_MODEL) else 0
    
    # Print system information
    print("="*60)
    print("Starting Training System")
    print("="*60)
    print(f"Save directory: {SUMMARY_DIR}")
    
    if os.path.exists(NN_MODEL):
        print(f"Found model: nn_model_ep_{LAST_MODEL_EPOCH}.pth")
        print(f"Will resume training from Epoch: {start_epoch}")
    else:
        print(f"Model not found: {NN_MODEL}")
        print(f"Starting training from scratch (Epoch 0)")
    
    print(f"Target total epochs: {TRAIN_EPOCH}")
    print("="*60)
    
    # Create communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # Start central coordinator process
    coordinator = mp.Process(
        target=central_agent,
        args=(net_params_queues, exp_queues)
    )
    coordinator.start()

    # Start local agent processes - PASS start_epoch (FIXED)
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(
            target=agent,
            args=(i, net_params_queues[i], exp_queues[i], start_epoch)
        ))
    
    for i in range(NUM_AGENTS):
        agents[i].start()

    # Wait for all processes to complete
    coordinator.join()

# ============================================================================
# SECTION 9: SCRIPT ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    main()
    
