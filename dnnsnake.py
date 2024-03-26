import pygame
import random
from collections import deque
import torch
import numpy as np
from DQNcnn import DQN, DQN_Agent, ReplayMemory
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import logging
import torch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    plt.ion()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
BOARD_SIZE = 11
CELL_SIZE = 40
GRID_COLOR = (200, 200, 200)
BACKGROUND_COLOR = (0, 0, 0)
SNAKE_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)
WINDOW_WIDTH = BOARD_SIZE * CELL_SIZE
WINDOW_HEIGHT = BOARD_SIZE * CELL_SIZE
DIRECTIONS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

# Global variable to track food locations
food_positions = []

def spawn_food(snakes,i):
    for i in range(i):
        # Generate a random position for new food
        new_food_pos = (random.randint(0, BOARD_SIZE-1), random.randint(0, BOARD_SIZE-1))
        
        # Check if the new food position is not on a snake or already a food position
        if not any(new_food_pos in snake.body for snake in snakes) and new_food_pos not in food_positions:
            food_positions.append(new_food_pos)
        
class Snake:
    def __init__(self, position,id):
        self.body = [position]  # Use a list for the snake's body
        self.direction = "RIGHT"
        self.grow_to = 3  # Initial size of the snake
        self.is_alive = True  # Keep track of whether the snake is alive
        self.health = 300  # Initialize the health value of the snake
        self.id = id
        self.score = 0
        self.previous_positions = deque([], maxlen=4)  # Store the last four positions
    def move(self, action_tensor, snakes):
        reward = 0
        if not self.is_alive:
            return -10, False, self.grow_to

        self.health -= 1
        if self.health <= 0:
            self.is_alive = False
            return -10, self.is_alive, self.grow_to

        action = action_tensor.item()  # Convert tensor to integer

        # Calculate the distance to the nearest food before the move
        head_x, head_y = self.body[0]

        try:
            prev_distance_to_food = min(abs(head_x - food[0]) + abs(head_y - food[1]) for food in food_positions)
        except ValueError as e:
            spawn_food(snakes,3)
            prev_distance_to_food = min(abs(head_x - food[0]) + abs(head_y - food[1]) for food in food_positions)
            logger.error(f"Error calculating prev_distance_to_food: {e}")
            logger.debug(f"Current food positions: {food_positions}")
            logger.debug(f"Current game state: {snakes}")
            

        direction_order = ["UP", "RIGHT", "DOWN", "LEFT"]
        if action == 1:  # Turn right
            new_dir_index = (direction_order.index(self.direction) + 1) % 4
        elif action == 2:  # Turn left
            new_dir_index = (direction_order.index(self.direction) - 1) % 4
        else:  # Go straight, action == 0
            new_dir_index = direction_order.index(self.direction)
        new_direction = direction_order[new_dir_index]

        # Calculate new head position based on the new direction
        dir_x, dir_y = DIRECTIONS[new_direction]
        new_head = (head_x + dir_x, head_y + dir_y)

        try:
            new_distance_to_food = min(abs(new_head[0] - food[0]) + abs(new_head[1] - food[1]) for food in food_positions)
        except ValueError as e:
            spawn_food(snakes,3)
            new_distance_to_food = min(abs(new_head[0] - food[0]) + abs(new_head[1] - food[1]) for food in food_positions)
            logger.error(f"Error calculating new_distance_to_food: {e}")
            logger.debug(f"Current food positions: {food_positions}")
            logger.debug(f"Current game state: {snakes}")
            raise
        # Give a reward if the distance to the nearest food decreases
        if new_distance_to_food < prev_distance_to_food:
            reward += 5

        # Check for moving in a square
        if new_head in self.previous_positions:
            reward -= 5

        # Check for collisions with walls
        if new_head[0] < 0 or new_head[0] >= BOARD_SIZE or new_head[1] < 0 or new_head[1] >= BOARD_SIZE:
            self.is_alive = False
            reward = -50  # Penalty for hitting a wall
            return reward, self.is_alive, self.grow_to

        # Check for collisions with enemy snakes' bodies
        for other_snake in snakes:
            if other_snake is not self:
                if new_head in other_snake.body:
                    self.is_alive = False
                    reward = -50  # Negative reward for colliding with an enemy snake's body
                    return reward, self.is_alive, self.grow_to
        # Check for collisions with own body
        if new_head in list(self.body)[1:]:
            self.is_alive = False
            reward = -50  # Negative reward for colliding with own body
            return reward, self.is_alive, self.grow_to
        # Check for food consumption
        if new_head in food_positions:
            self.grow_to += 1
            food_positions.remove(new_head)
            self.health = 300
            self.score += 1
            reward = 100  # Positive reward for eating food
            spawn_food(snakes,1)
        else:
            # If not growing, remove the tail
            if len(self.body) > self.grow_to:
                self.body.pop()

        # If all checks pass, move forward
        self.body.insert(0, new_head)  # Insert new head at the beginning
        self.direction = new_direction  # Update the direction based on action

        # Update previous positions list with the new head position
        self.previous_positions.append(new_head)

        return reward, self.is_alive, self.score

    def draw(self,screen):
        for index, segment in enumerate(self.body):
            x, y = segment
            # Check if this segment is the head
            if index == 0:
                # Use a distinct color for the head
                head_color = (0, 0, 255)  # Example: Red
                pygame.draw.rect(screen, head_color, pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                # Optionally, draw a smaller inner rectangle for the head as well, using the same or a slightly different color
                pygame.draw.rect(screen, (0, 100, 255), pygame.Rect(x * CELL_SIZE + 4, y * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8))
            else:
                # Draw the body segment
                pygame.draw.rect(screen, SNAKE_COLOR, pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                # Draw a smaller inner rectangle for visual differentiation
                pygame.draw.rect(screen, (0, 200, 0), pygame.Rect(x * CELL_SIZE + 4, y * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8))

def update_grid(screen):
    for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)
    for food_pos in food_positions:
        rect = pygame.Rect(food_pos[0] * CELL_SIZE, food_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, FOOD_COLOR, rect)

def reset_game():
    global snakes, food_positions
    # Reset snakes to their starting positions and directions
    snakes = [Snake((5, 5),0), Snake((7, 7),1)]
    for snake in snakes:
        snake.is_alive = True
        snake.body = deque([(snake.body[0][0], snake.body[0][1])])
        snake.grow_to = 3

    # Clear existing food and spawn new food
    food_positions.clear()
    spawn_food(snakes,3)
def get_matrix_state(snakes, snake_id):
    # Initialize an 11x11 matrix with zeros for the game grid
    matrix_state = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    # Initialize an 11x11 matrix with zeros for the distance to the nearest food
    distance_to_food_map = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    # Initialize an 11x11 matrix with zeros for the distance to the nearest enemy snake
    distance_to_enemy_snake_map = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    # Mark food positions with 1 in the game grid matrix
    for food in food_positions:
        matrix_state[food[1]][food[0]] = 1
    
    # Calculate the distance to the nearest food for each cell
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            min_distance = float('inf')
            for food in food_positions:
                distance = abs(i - food[1]) + abs(j - food[0])  # Manhattan distance
                min_distance = min(min_distance, distance)
            distance_to_food_map[i][j] = min_distance

    max_food_distance = max(max(row) for row in distance_to_food_map)
    if max_food_distance != 0:
        distance_to_food_map = [[round(cell / max_food_distance, 3) for cell in row] for row in distance_to_food_map]
 
    # Calculate the distance to the nearest enemy snake for each cell
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            min_distance = float('inf')
            for snake in snakes:
                if snake.id != snake_id:  # Only consider enemy snakes
                    for part in snake.body:
                        distance = abs(i - part[1]) + abs(j - part[0]) # Manhattan distance
                        min_distance = min(min_distance, distance)
            distance_to_enemy_snake_map[i][j] = min_distance

    max_enemy_distance = max(max(row) for row in distance_to_enemy_snake_map)
    if max_enemy_distance != 0:
        distance_to_enemy_snake_map = [[round(cell / max_enemy_distance, 3) for cell in row] for row in distance_to_enemy_snake_map]
 
    # Iterate over each snake and mark their positions in the game grid matrix
    for idx, snake in enumerate(snakes):
        for part in snake.body:
            x, y = part
            if snake.id == snake_id:
                if part == snake.body[0]:
                    matrix_state[y][x] = 3  # Snake head
                else:
                    matrix_state[y][x] = 2  # Snake body
            else:
                if part == snake.body[0]:
                    matrix_state[y][x] = 5  # Other snake head
                else:
                    matrix_state[y][x] = 4  # Other snake body
    
    # Stack the game grid matrix, distance to food map, and distance to enemy snake map along a new axis
    stacked_state = np.stack((matrix_state, distance_to_food_map, distance_to_enemy_snake_map), axis=0)
    
    # print("_____________________________")
    # for state in distance_to_enemy_snake_map:
    #     print(state)
    return stacked_state
def main(args):
    model_paths = args.model
    display_on = args.displayOn
    speed = args.speed
    if display_on:
        pygame.init()
        # Set up the display
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('BattleSnake Game Board')

    plot_mean_scores, total_score, record, score, reward = [[], []], [0, 0], [0, 0], [0, 0], [0, 0]
    running = True
    games = 0
    
    global snakes, agents  # Make sure these are accessible and modifiable
    #clock = pygame.time.Clock()
    snakes = [Snake((5, 5),0), Snake((7, 7),1)]  # Example snakes
    agents = [DQN_Agent(n_actions=3, n_observations=11*11, input_channels=3) for _ in snakes]
    spawn_food(snakes,3)

    window_size = 300
    window_scores = [[] for _ in snakes]

    while running:
        plot_scores = [[], []]
        if display_on:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        
        if len(food_positions) == 0:
            spawn_food(snakes,3)
        all_snakes_dead = all(not snake.is_alive for snake in snakes)  # Check if all snakes are dead
        
        # Modify the plotting section inside the main loop
        if all_snakes_dead:
            games +=1
            for i, snake in enumerate(snakes):
                if score[i] > record[i]:
                    record[i] = score[i]
                    # save the weights when new best score is reached
                    torch.save(agents[i].policy_net.state_dict(), f'best_model_snake_{i}.pth')

                total_score[i] += score[i]
                window_scores[i].append(score[i])  # Append the score to the window scores list
                
            if games % window_size == 0:  # Check if 100 turns have passed
                print(sum(window_scores[0]) / 300," len: ",len(window_scores[0]))
                print(sum(window_scores[1]) / 300," len: ",len(window_scores[1]))

                print(record)
                print("======")
                #plot_mean_scores[i].append(window_mean_score)
                window_scores = [[] for _ in snakes]  # Clear the window scores list for the next window
                games = 0
                print("<><><><><><")
            # Plotting
            # if games % window_size == 0:  # Plot only every 100 turns
            #     plt.clf()  # Clear the current figure
            #     for i in range(len(snakes)):
            #         plt.plot(plot_mean_scores[i], label=f'Snake {i+1} Mean Score')
            #     plt.xlabel('Window (100 turns)')
            #     plt.ylabel('Mean Score')
            #     plt.title('Mean Scores Over Time')
            #     plt.legend()
            #     plt.draw()
            #     plt.pause(0.001)  # Pause to ensure the plot gets updated
            
            reset_game()

        for i, snake in enumerate(snakes):
            if snake.is_alive:
                old_status = get_matrix_state(snakes,i)
                action = agents[i].select_action(old_status)
                reward[i], dead, score[i] = snake.move(action, snakes)

                if dead:
                    #plot_mean_scores[i].append(score[i])
                    new_status = None
                else:
                    new_status = get_matrix_state(snakes,i)

                agents[i].memory.push(old_status, action,new_status,reward[i])
                agents[i].optimize_model()
                
                target_net_state_dict = agents[i].target_net.state_dict()
                policy_net_state_dict = agents[i].policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*agents[i].TAU + target_net_state_dict[key]*(1-agents[i].TAU)
                agents[i].target_net.load_state_dict(target_net_state_dict)
        if display_on:
            screen.fill(BACKGROUND_COLOR)
            update_grid(screen)
            for snake in snakes:
                if snake.is_alive:
                    snake.draw(screen)
                pygame.display.flip()
    pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BattleSnake Game with AI Agents")
    parser.add_argument('--model', nargs='+', help='Path to the pre-trained model file(s). Accepts up to two model file paths/names.', default=[])
    parser.add_argument('--displayOn', action='store_true', help='Enable game display.')
    #currently does nothing but thought might be useful later for showing trained snake movement
    parser.add_argument('--speed', type=int, default=1, help='Time between steps in milliseconds.')
    args = parser.parse_args()
    main(args)
