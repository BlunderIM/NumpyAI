import numpy as np
import os
from sarsa import SarsaLearner
from utils import get_bot_position, get_target_position, control_bot, discretize, interpret_arrow, plot_maze


if __name__ == "__main__":
    plot_journey = False
    epochs = 50
    # Initialize the learner
    learner = SarsaLearner(num_states=100,  num_actions=4,  alpha=0.2,  gamma=0.9,  rar=0.98, radr=0.999)

    # Load the maze
    maze = np.genfromtxt('../data/maze.csv', delimiter=',')
    start_position = get_bot_position(maze)
    goal_position = get_target_position(maze)
    action_list, position_list, reward_list, rar_list, count_list, epoch_list, slip_list = [], [], [], [], [], [], []
    epoch_complete_list = []

    for ep in range(1, epochs + 1):
        total_reward = 0
        data = maze.copy()
        bot_position = start_position
        position_list.append(bot_position)
        state = discretize(bot_position)
        # Learner performs an action
        action = learner.query_no_update(state)
        action_list.append(action)
        reward_list.append(0)
        count = 0
        epoch_list.append(ep)
        count_list.append(count)
        epoch_complete_list.append(False)
        # Loop to define convergence
        while (bot_position != goal_position) & (count < 10000):
            new_position, step_reward, slip = control_bot(data, bot_position, action)
            if new_position == goal_position:
                epoch_complete_list.append(True)
                reward = 1
            else:
                epoch_complete_list.append(False)
                reward = step_reward
            reward_list.append(reward)
            state = discretize(new_position)
            action, rar = learner.query(state, reward)
            action_list.append(action)
            rar_list.append(rar)
            slip_list.append(slip)

            bot_position = new_position
            position_list.append(bot_position)

            total_reward += step_reward
            count += 1
            count_list.append(count)
            epoch_list.append(ep)

        print(f"### Epoch {ep} complete ###")
        slip_list.append(False)

    # plotting
    if plot_journey:
        if not os.path.exists("../images"):
            os.makedirs("../images")

        for idx in range(len(count_list)):
            bot_position = position_list[idx]
            slip_condition = slip_list[idx]
            epoch_complete = epoch_complete_list[idx]
            current_action = action_list[idx]
            delta_x, delta_y = interpret_arrow(current_action)
            recent_rar = rar_list[:idx+1]
            recent_award = reward_list[:idx+1]
            ep = epoch_list[idx]
            count = count_list[idx]

            if len(str(idx)) == 1:
                fig_title = '../images/seq_000' + str(idx) + '.png'
            elif len(str(idx)) == 2:
                fig_title = '../images/seq_00' + str(idx) + '.png'
            elif len(str(idx)) == 3:
                fig_title = '../images/seq_0' + str(idx) + '.png'
            else:
                fig_title = '../images/seq_' + str(idx) + '.png'
            plot_maze(maze, bot_position, current_action, delta_x, delta_y, recent_rar, recent_award, fig_title, ep,
                      count, rar_list, reward_list, slip_condition, epoch_complete)
