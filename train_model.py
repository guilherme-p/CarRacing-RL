import argparse
import gym
from collections import deque
from DQNAgent import DQNAgent
from helper_functions import state2gray, generate_input
import datetime

RENDER                        = True
STARTING_EPISODE              = 1
ENDING_EPISODE                = 1000
EPSILON                       = 1
SKIP_FRAMES                   = 5
TRAINING_BATCH_SIZE           = 64
SAVE_TRAINING_FREQUENCY       = 5
UPDATE_TARGET_MODEL_FREQUENCY = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training a DQN agent to play CarRacing.")
    parser.add_argument("-m", "--model", help="Specify the last trained model path if you want to continue training after it.")
    parser.add_argument("-r", "--record", help="Specify if you want to record the training into ./videos/ folder", action="store_true")
    args = parser.parse_args()

    env = gym.make("CarRacing-v0")

    if args.record:
        from gym.wrappers.monitor import Monitor
        env = Monitor(env, "./videos", video_callable=False, force=True)
    
    agent = DQNAgent(epsilon=EPSILON)

    if args.model:
        agent.load(args.model)

    for e in range(STARTING_EPISODE, ENDING_EPISODE+1):
        print("{}: Starting episode {}/{}".format(datetime.datetime.now(), e, ENDING_EPISODE), flush=True)

        init_state = env.reset()
        init_state = state2gray(init_state)

        total_reward = 0
        state_frame_stack = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False
        
        while True:
            if RENDER:
                env.render()

            current_state_frame_stack = generate_input(state_frame_stack)
            action = agent.act(current_state_frame_stack)

            reward = 0
            for _ in range(SKIP_FRAMES+1):
                next_state, r, done, info = env.step(action)
                reward += r

                if done:
                    break

            # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 50 and reward < 0 else 0

            # Extra bonus for the model if it uses full gas
            if action[1] == 1 and action[2] == 0:
                reward *= 1.5

            total_reward += reward

            next_state = state2gray(next_state)
            state_frame_stack.append(next_state)
            next_state_frame_stack = generate_input(state_frame_stack)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or negative_reward_counter >= 25 or total_reward < 0:
                print("Episode: {}/{}, Time Frames: {}, Total Rewards (adjusted): {:.2}, Epsilon: {:.2}".format(e, 
                                                                                                                ENDING_EPISODE, 
                                                                                                                time_frame_counter, 
                                                                                                                float(total_reward), 
                                                                                                                float(agent.epsilon))); break

            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)

            time_frame_counter += 1

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            agent.save("./save/train{}_{}.h5".format(agent.version, e))

    env.close()
