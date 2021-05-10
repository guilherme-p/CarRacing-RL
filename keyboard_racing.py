import gym
import numpy as np


if __name__ == '__main__':
    gym.envs.register(
        id='CarRacing-v1', # CHANGED
        entry_point='gym.envs.box2d:CarRacing',
        max_episode_steps=2000, # CHANGED FROM 1000
        reward_threshold=900,
    )

    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart

        if k == key.ESCAPE:
            restart = True

        if k == key.A:
            a[0] = -1.0

        if k == key.D:
            a[0] = +1.0

        if k == key.W:
            a[1] = +1.0

        if k == key.S:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.A and a[0] == -1.0:
            a[0] = 0

        if k == key.D and a[0] == +1.0:
            a[0] = 0

        if k == key.W:
            a[1] = 0
            
        if k == key.S:
            a[2] = 0

    env = gym.make('CarRacing-v1')
    env.render()

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    record_video = False

    if record_video:
        from gym.wrappers.monitor import Monitor
        env = Monitor(env, "/tmp/video-test", force=True)

    isopen = True

    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False

        while True:
            s, r, done, info = env.step(a)
            total_reward += r

            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))

            steps += 1
            isopen = env.render()

            if done or restart or isopen == False:
                break
            
    env.close()