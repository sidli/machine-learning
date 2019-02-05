import numpy as np
import copy

def main():
    reward = [[0, 1, 0.5, 0.5], [0.5, 0, 1, 0.5], [-1, 0.5, 0, 0.5], [-1, 0.5, 0.5, 0]]
    gamma = 0.8
    transform = lambda state,action: action
    values = [0.0] * 4
    values_action = [-1] * 4
    action_stack = []
    for times in range(10):
        while True:
            values_old = copy.copy(values)
            action_stack.append(copy.copy(values_action))
            for state in range(4):
                rand_values = []
                for action in range(4):
                    value_tmp = reward[state][action] + gamma * values[transform(state,action)]
                    if value_tmp < 0.0:
                        value_tmp = 0.0
                    rand_values.append(value_tmp)
                rand_values = np.array(rand_values)
                p = rand_values/np.sum(rand_values)
                action = np.random.choice([0, 1, 2, 3], p = p.ravel())
                values_action[state] = action
                values[state] = rand_values[action]
            if np.sum(np.array(values) - np.array(values_old)) <= 0.001:
                break
        print(values)
        print(values_action)
        print(action_stack)

if __name__ == "__main__":
    main()
