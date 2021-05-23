import argparse
import gym
from collections import deque
from DQNAgent import DQNAgent
from helper_functions import state2gray, generate_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play CarRacing by the trained model.")
    parser.add_argument("-m", "--model", required=True, help="The `.h5` file of the trained model.")
    parser.add_argument("-e", "--episodes", type=int, default=1, help="The number of episodes the model should play.")
    args = parser.parse_args()

    env = gym.make("CarRacing-v0")
    agent = DQNAgent(epsilon=0) # Set epsilon to 0 to ensure all actions are performed by the agent
    agent.load(args.model)

    for e in range(args.episodes):
        init_state = env.reset()
        init_state = state2gray(init_state)

        total_reward = 0
        state_frame_stack = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        
        while True:
            env.render()

            current_state_frame_stack = generate_input(state_frame_stack)
            action = agent.act(current_state_frame_stack)
            next_state, reward, done, info = env.step(action)

            total_reward += reward

            next_state = state2gray(next_state)
            state_frame_stack.append(next_state)

            if done:
                print("Episode: {}/{}, Time Frames: {}, Total Rewards: {:.2}".format(e + 1, 
                                                                                     args.episodes, 
                                                                                     time_frame_counter, 
                                                                                     float(total_reward))); break

            time_frame_counter += 1
