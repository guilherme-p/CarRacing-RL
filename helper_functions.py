import numpy as np
import cv2

def state2gray(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

    #Normalize
    state = state.astype(float)
    state /= 255.0
    
    return state

def generate_input(deque):
    frame_stack = np.array(deque)
    # Transpose stack (stack_len, x, y) -> (x, y, stack_len) e.g. (3, 96, 96) -> (96, 96, 3)
    return np.transpose(frame_stack, (1, 2, 0))