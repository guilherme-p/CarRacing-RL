import gym
import numpy as np


if __name__ == "__main__":
    a = np.array([0.0, 0.5, 0.0]) #Steering, gas, breaks

    env = gym.make("CarRacing-v0")
    env.render()

    is_open = True

    init_state = env.reset()
    negative_reward_counter = 0
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        s, r, done, info = env.step(a)
        total_reward += r

        a = np.array([0.0, 0.5, 0.0])

        if steps % 200 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        
        steps += 1
        is_open = env.render()

        if done or is_open == False:
            break

        negative_reward_counter = negative_reward_counter + 1 if r < 0 else 0

        if negative_reward_counter >= 20:
            a[0] = np.random.choice([-1.0, +1.0])

            for _ in range(4):
                s, r, done, info = env.step(a)
                total_reward += r

                steps += 1
                is_open = env.render()

                if done or is_open == False:
                    break

            negative_reward_counter = 0
     
    print("step {} total_reward {:+0.2f}".format(steps, total_reward))

    env.close()