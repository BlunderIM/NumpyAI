import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.image as mpimg


def get_bot_position(maze):
    # Set row and col positions
    pos_r, pos_c = -1, -1
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if maze[r, c] == 2:
                pos_r = r
                pos_c = c
    return pos_r, pos_c


def get_target_position(maze):
    pos_r, pos_c = -1, -1
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if maze[r, c] == 3:
                pos_r = r
                pos_c = c
    return pos_r, pos_c


def control_bot(maze, old_pos, a):
    # north = 0, east = 1, south = 2, west = 3

    old_r, old_c = old_pos
    random_rate = 0.2
    trap_reward = -150
    goal_reward = 1
    slip = False

    if np.random.uniform(0, 1) <= random_rate:
        pre_determined_action = a
        a = np.random.choice(4, 1)[0]
        if a != pre_determined_action:
            slip = True

    # Updating the location
    if a == 0:
        new_r = old_r - 1
        new_c = old_c
    if a == 1:
        new_c = old_c + 1
        new_r = old_r
    if a == 2:
        new_r = old_r + 1
        new_c = old_c
    if a == 3:
        new_c = old_c - 1
        new_r = old_r

    # Reward for any move is -1
    reward = -1

    # If move is legal, allow it. If not legal, stay in same position

    if new_r < 0:  # out of grid
        new_r = old_r
        rew_c = old_c
    elif new_r >= maze.shape[0]:  # out of grid
        new_r = old_r
        new_c = old_c
    elif new_c < 0:  # out of grid
        new_c = old_c
        new_r = old_r
    elif new_c >= maze.shape[1]:
        new_c = old_c
        new_r = old_r
    elif maze[new_r, new_c] == 1:  # this is a wall
        new_r = old_r
        new_c = old_c
    elif maze[new_r, new_c] == 5:  # if this is a trap
        reward = trap_reward
    elif maze[new_r, new_c] == 3:  # this is the target
        reward = 1

    # Returning the new location and the reward
    return (new_r, new_c), reward, slip


def discretize(pos):
    # Converting a position to a state
    return pos[0] * 10 + pos[1]


def interpret_arrow(current_action):
    # north = 0, east = 1, south = 2, west = 3
    if current_action == 0:
        delta_x = -0.5
        delta_y = 0
    elif current_action == 1:
        delta_x = 0
        delta_y = 0.5
    elif current_action == 2:
        delta_x = 0.5
        delta_y = 0
    elif current_action == 3:
        delta_x = 0
        delta_y = -0.5
    return delta_x, delta_y


def plot_maze(maze, bot_position, current_action, delta_x, delta_y, recent_rar, recent_award, fig_title, epoch, count,
              rar_list, reward_list, slip_condition=False, epoch_complete=False):
    fig = plt.figure(constrained_layout=True, figsize=(15, 8));
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax1.plot([0], [0], color="white")
    ax1.set_xlim([-1, 11])
    ax1.set_ylim([-1, 11])
    ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax1.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])
    ax1.axis('off')
    [ax1.axhline(y=i, xmin=0.085, xmax=0.915, color="gray", zorder=1) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    [ax1.axvline(x=i, ymin=0.085, ymax=0.915, color="gray", zorder=1) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    ax1.add_patch(Rectangle((0, 0), 10, 10, color="gray", fill=False))
    ax1title = '\n Epoch #' + str(epoch) + ' Step #' + str(count)
    ax1.set_title(ax1title, fontsize=30, fontweight='bold', color='black', loc='center')

    if slip_condition:
        box_x, box_y = 4, 10.2
        slip_box = patches.Rectangle((box_x, box_y), 2, 0.8, linewidth=1, edgecolor='purple', facecolor='purple', clip_on=False)
        ax1.add_patch(slip_box)
        text_x = box_x +1 # X-coordinate for the text (center of the box)
        text_y = box_y +0.4  # Y-coordinate for the text (center of the box)
        ax1.text(text_x, text_y, 'slip', fontsize=20, fontweight='bold', color='white',
                ha='center', va='center')

    for i in range(10):
        for j in range(10):
            element = maze[i, j]
            if element == 1:
                ax1.add_patch(Rectangle((i, j), 1, 1, color="black", zorder=1))
            if element == 3:
                ax1.add_patch(Rectangle((i, j), 1, 1, color="green", zorder=1))
            if element == 5:
                ax1.add_patch(Rectangle((i, j), 1, 1, color="red", zorder=1))

    # Bot
    image_path = '../data/bot.jpg'
    image = mpimg.imread(image_path)
    image_size = 0.7
    ax1.imshow(image, extent=(bot_position[0] + 0.5 - image_size/2, bot_position[0] + 0.5 + image_size/2,
                              bot_position[1] + 0.5 - image_size/2, bot_position[1] + 0.5 + image_size/2), zorder=11)
    if not epoch_complete:
        ax1.arrow(x=bot_position[0] + 0.5, y=bot_position[1] + 0.5, dx=delta_x, dy=delta_y, width=0.1,
                  color="orange", zorder=10)
    direction_patch = mpatches.Patch(color='orange', label='Intended Direction')
    target_patch = mpatches.Patch(color='green', label='Target')
    trap_patch = mpatches.Patch(color='red', label='Trap')
    wall_patch = mpatches.Patch(color='black', label='Wall')
    ax1.legend(handles=[direction_patch, target_patch, trap_patch, wall_patch], bbox_to_anchor=(0.93, 0.0),
               ncol=6, fontsize=10)
    ax1.set_facecolor('white')

    ax2.plot(recent_rar, color='royalblue')
    ax2.set_xlim([0, len(rar_list)])
    ax2.set_ylim([0, 1])
    ax2.set_title('Exploration vs Exploitation - Random Action Rate', fontsize=15, fontweight='bold', color='black',
                  loc='center')
    ax2.set(ylabel="Probability of Random Action")
    ax2.set(xlabel="Step Count")
    ax2.set_facecolor('white')

    ax3.plot(np.cumsum(recent_award), color='r')
    ax3.set_xlim([0, len(reward_list)])
    ax3.set_ylim([0, np.cumsum(reward_list)[-1]])
    ax3.invert_yaxis()
    ax3.set_title('Cumulative Negative reward', fontsize=15, fontweight='bold', color='black', loc='center')
    ax3.set(ylabel="Reward Points")
    ax3.set(xlabel="Step Count")
    ax3.set_facecolor('white')
    fig.set_facecolor('white')

    fig.savefig(fig_title)
    plt.close(fig)
