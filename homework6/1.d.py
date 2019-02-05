import numpy as np
import copy

def main():
    reward = [[0, 1, 0.5, 0.5], [0.5, 0, 1, 0.5], [-1, 0.5, 0, 0.5], [-1, 0.5, 0.5, 0]]
    gamma = 0.01
    transform = lambda state,action: action
    values = [0.0] * 4
    values_action = [-1] * 4
    action_stack = []
    while True:
        values_old = copy.copy(values)
        action_stack.append(copy.copy(values_action))
        for state in range(4):
            max_value = 0.0
            for action in range(4):
                value_tmp = reward[state][action] + gamma * values[transform(state,action)]
                if value_tmp > max_value:
                    max_value = value_tmp
                    values_action[state] = action
            values[state] = max_value
        if np.sum(np.array(values) - np.array(values_old)) == 0:
            break
    print(values)
    print(values_action)
    print(action_stack)

if __name__ == "__main__":
    main()
