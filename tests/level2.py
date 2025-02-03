import pygame
import sys
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque, namedtuple

############################################
# 0) GLOBAL CONFIG & CONSTANTS
############################################
GRID_SIZE = 15
CELL_SIZE = 40
INFO_PANEL_WIDTH = 300
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + INFO_PANEL_WIDTH
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 10

# Colors
BLACK = (0, 0, 0)
ORANGE = (255, 140, 0)
GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
EYE_COLOR = (255, 255, 255)
PUPIL_COLOR = (0, 0, 0)
RED = (255, 0, 0)
GREY = (80, 80, 80)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)

# For 22-input Snake
STATE_DIM = 22
ACTION_DIM = 3

MODEL_CHECKPOINT = "models/best_dqn_snake_level2.pth"
UNDO_STACK_SIZE = 200

# Initialize food image globally
food_img = None

############################################
# 1) DQN NETWORK (22 -> 3)
############################################
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

############################################
# 2) SNAKE ENV (Level 2)
############################################
class SnakeGame22:
    DIRECTIONS_8 = [(-1,0), (-1,1), (0,1), (1,1),
                    (1,0), (1,-1), (0,-1), (-1,-1)]
    
    LEFT_TURN = {
        (-1,0): (0,-1),  # up -> left
        (1,0): (0,1),    # down -> right
        (0,-1): (1,0),   # left -> down
        (0,1): (-1,0)    # right -> up
    }
    RIGHT_TURN = {
        (-1,0): (0,1),   # up -> right
        (1,0): (0,-1),   # down -> left
        (0,-1): (-1,0),  # left -> up
        (0,1): (1,0)     # right -> down
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(GRID_SIZE//2, GRID_SIZE//2)]
        self.direction = (-1, 0)  # starting direction: up
        self.food = self._place_food()
        self.done = False
        self.steps = 0
        self.cumulative_reward = 0.0
        return self._get_state()

    def _place_food(self):
        while True:
            pos = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE))
            if pos not in self.snake:
                return pos

    def _distance_to_wall(self, hx, hy, dx, dy):
        dist = 0
        cx, cy = hx, hy
        while 0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE:
            cx += dx
            cy += dy
            dist += 1
        return dist

    def _distance_to_body(self, hx, hy, dx, dy):
        dist = 0
        cx, cy = hx, hy
        while True:
            cx += dx
            cy += dy
            dist += 1
            if not (0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE):
                return dist
            if (cx, cy) in self.snake:
                return dist

    def _get_food_direction_bits(self, hx, hy):
        # We can interpret bits as: [food_up, food_down, food_left, food_right]
        fx, fy = self.food
        return [
            int(fx < hx),  # food above?
            int(fx > hx),  # food below?
            int(fy < hy),  # food left?
            int(fy > hy)   # food right?
        ]

    def _get_state(self):
        hx, hy = self.snake[0]
        body_dists = [self._distance_to_body(hx, hy, dx, dy) for dx, dy in self.DIRECTIONS_8]
        wall_dists = [self._distance_to_wall(hx, hy, dx, dy) for dx, dy in self.DIRECTIONS_8]
        food_bits = self._get_food_direction_bits(hx, hy)
        dir_vec = list(self.direction)
        return np.array(body_dists + wall_dists + food_bits + dir_vec, dtype=np.float32)

    def step(self, action):
        """action: 0 -> left, 1 -> straight, 2 -> right"""
        if self.done:
            return self._get_state(), 0.0, True

        # Simple shaping to encourage turning vs. going straight
        # so we see different states. Adjust to your preference.
        turn_reward = 0.1 if action == 1 else -1.0

        if action == 0:   # turn left
            self.direction = self.LEFT_TURN[self.direction]
        elif action == 2: # turn right
            self.direction = self.RIGHT_TURN[self.direction]

        hx, hy = self.snake[0]
        dx, dy = self.direction
        new_head = (hx + dx, hy + dy)

        # Check collision with wall
        if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE):
            self.done = True
            reward = -10.0 + turn_reward
            self.cumulative_reward += reward
            return self._get_state(), reward, True

        # Check collision with self
        if new_head in self.snake:
            self.done = True
            reward = -15.0 + turn_reward
            self.cumulative_reward += reward
            return self._get_state(), reward, True

        # Move snake
        self.snake.insert(0, new_head)
        reward = -0.05 + turn_reward

        # Check if got food
        if new_head == self.food:
            reward += 15.0
            self.food = self._place_food()
        else:
            self.snake.pop()

        self.steps += 1
        self.cumulative_reward += reward
        return self._get_state(), reward, False

    def snapshot(self):
        return {
            "snake": list(self.snake),
            "direction": self.direction,
            "food": self.food,
            "done": self.done,
            "steps": self.steps,
            "cumulative_reward": self.cumulative_reward
        }

    def restore(self, snap):
        self.snake = list(snap["snake"])
        self.direction = snap["direction"]
        self.food = snap["food"]
        self.done = snap["done"]
        self.steps = snap["steps"]
        self.cumulative_reward = snap["cumulative_reward"]

############################################
# 3) LOAD PRETRAINED MODEL
############################################
device = torch.device("cpu")
model = DQN(STATE_DIM, ACTION_DIM).to(device)
model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
model.eval()

############################################
# 4) PYGAME VISUALIZATION
############################################
def init_pygame():
    pygame.init()
    pygame.display.set_caption("AI Snake with Cartoon Rendering")
    return pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)

def draw_visited_cells(surface, visited_positions):
    """Draw the last 50 visited cells in gray."""
    for (r, c) in visited_positions:
        rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, GREY, rect)

def draw_grid(surface):
    for x in range(GRID_SIZE):
        pygame.draw.line(surface, (60,60,60), (x*CELL_SIZE, 0), (x*CELL_SIZE, GRID_SIZE*CELL_SIZE))
    for y in range(GRID_SIZE):
        pygame.draw.line(surface, (60,60,60), (0, y*CELL_SIZE), (GRID_SIZE*CELL_SIZE, y*CELL_SIZE))

def draw_snake(surface, snake, tongue_visible):
    # Draw body segments with tapering
    for i, (r, c) in enumerate(snake):
        center_x = c * CELL_SIZE + CELL_SIZE//2
        center_y = r * CELL_SIZE + CELL_SIZE//2
        segment_size = int(CELL_SIZE * (1 - i/len(snake)))
        radius = max(10, segment_size//2)
        
        pygame.draw.circle(surface, ORANGE, (center_x, center_y), radius)
        # Optional: small green circle "patterns" every 5 segments
        if i % 5 == 0:
            pygame.draw.circle(surface, GREEN, (center_x, center_y), max(2, radius//4))

    # Draw head with face
    if snake:
        head_r, head_c = snake[0]
        head_rect = (head_c*CELL_SIZE, head_r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.ellipse(surface, ORANGE, head_rect)

        # Eyes
        eye_left = (head_c*CELL_SIZE + 10, head_r*CELL_SIZE + 10)
        eye_right = (head_c*CELL_SIZE + 30, head_r*CELL_SIZE + 10)
        pygame.draw.circle(surface, WHITE, eye_left, 6)
        pygame.draw.circle(surface, WHITE, eye_right, 6)
        pygame.draw.circle(surface, PUPIL_COLOR, eye_left, 3)
        pygame.draw.circle(surface, PUPIL_COLOR, eye_right, 3)

        # Tongue
        if tongue_visible:
            start = (head_c*CELL_SIZE + 20, head_r*CELL_SIZE + 30)
            end = (head_c*CELL_SIZE + 20, head_r*CELL_SIZE + 40)
            pygame.draw.line(surface, RED, start, end, 3)

def draw_food(surface, food_pos):
    x = food_pos[1] * CELL_SIZE
    y = food_pos[0] * CELL_SIZE
    surface.blit(food_img, (x, y))

def draw_raycast_lines_to_wall(surface, game):
    """Draw lines from the snake's head to the wall boundaries (cyan)."""
    head_r, head_c = game.snake[0]
    head_center = (head_c*CELL_SIZE + CELL_SIZE//2, head_r*CELL_SIZE + CELL_SIZE//2)
    for dx, dy in game.DIRECTIONS_8:
        cx, cy = head_r, head_c
        while 0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE:
            cx += dx
            cy += dy
        end = (cy*CELL_SIZE + CELL_SIZE//2, cx*CELL_SIZE + CELL_SIZE//2)
        pygame.draw.line(surface, CYAN, head_center, end, 2)

def draw_raycast_lines_to_body(surface, game):
    """
    Draw lines from the snake's head to the first body-collision cell in each direction (magenta).
    """
    head_r, head_c = game.snake[0]
    head_center = (head_c*CELL_SIZE + CELL_SIZE//2, head_r*CELL_SIZE + CELL_SIZE//2)
    for dx, dy in game.DIRECTIONS_8:
        dist = game._distance_to_body(head_r, head_c, dx, dy)
        # The collision cell = head + dist*(dx, dy)
        # But game._distance_to_body() counts the head cell as step 0, 
        # so the collision cell is (head + (dist)*direction).
        collision_r = head_r + dx * dist
        collision_c = head_c + dy * dist
        collision_center = (collision_c*CELL_SIZE + CELL_SIZE//2, collision_r*CELL_SIZE + CELL_SIZE//2)
        pygame.draw.line(surface, MAGENTA, head_center, collision_center, 2)

def draw_info_panel(surface, game, state, last_reward, action):
    font = pygame.font.SysFont("Consolas", 18)
    panel = pygame.Rect(GRID_SIZE*CELL_SIZE, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(surface, (30,30,30), panel)
    
    y = 10
    def text(line, color=WHITE):
        nonlocal y
        surface.blit(font.render(line, True, color), (GRID_SIZE*CELL_SIZE+10, y))
        y += 22
    
    # Basic info
    text(f"Steps: {game.steps}")
    text(f"Length: {len(game.snake)}")
    text(f"Total Reward: {game.cumulative_reward:.1f}")
    text(f"Last Reward: {last_reward:.1f}")
    if action is not None:
        text(f"Action: {['Left','Straight','Right'][action]}", (255,215,0))
    else:
        text("Action: None", (255,215,0))

    # Show each part of the 22-dim state
    # Indices: 0..7 (body distances), 8..15 (wall distances), 16..19 (food bits), 20..21 (dir)
    text("-- State Breakdown --", CYAN)
    body_dists = state[0:8]
    wall_dists = state[8:16]
    food_bits = state[16:20]
    dir_bits = state[20:22]
    
    text("BodyDist(8):")
    text(str(body_dists.round(2).tolist()))

    text("WallDist(8):")
    text(str(wall_dists.round(2).tolist()))

    text("FoodBits(4):")
    text(str(food_bits.astype(int).tolist()))

    text("Direction(2):")
    text(str(dir_bits.round(2).tolist()))
    
############################################
# 5) MAIN LOOP
############################################
def main():
    global food_img
    screen = init_pygame()
    clock = pygame.time.Clock()
    
    # Load assets
    food_img_raw = pygame.image.load("food.png")
    food_img_scaled = pygame.transform.scale(food_img_raw, (CELL_SIZE, CELL_SIZE))
    food_img = food_img_scaled
    
    # Game setup
    game = SnakeGame22()
    state = game._get_state()
    last_reward = 0.0
    predicted_action = None
    history = deque(maxlen=UNDO_STACK_SIZE)
    
    # Keep track of the last 50 visited positions
    visited_positions = deque(maxlen=50)
    visited_positions.append(game.snake[0])  # initial head

    # Tongue animation
    tongue_visible = False
    tongue_timer = random.randint(1000, 3000)
    last_tongue = pygame.time.get_ticks()

    running = True
    while running:
        current_time = pygame.time.get_ticks()
        if current_time - last_tongue > tongue_timer:
            tongue_visible = not tongue_visible
            last_tongue = current_time
            tongue_timer = random.randint(500, 2500)
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RIGHT:  # Step forward
                    if not game.done:
                        history.append((game.snapshot(), state.copy(), last_reward))
                        with torch.no_grad():
                            q = model(torch.FloatTensor(state).unsqueeze(0))
                        predicted_action = q.argmax().item()
                        state, last_reward, done_flag = game.step(predicted_action)
                        # Record visited cell
                        visited_positions.append(game.snake[0])
                elif event.key == pygame.K_LEFT:  # Undo
                    if len(history) > 1:
                        history.pop()  # discard current
                        snap, old_state, old_reward = history[-1]
                        game.restore(snap)
                        state = old_state
                        last_reward = old_reward
                        predicted_action = None
                        # Rebuild visited_positions from scratch if you want perfect accuracy,
                        # or just let the visited positions keep their memory. For simplicity,
                        # we'll keep them as is. (Optional: you could track them in snap too.)
        
        # Rendering
        screen.fill(BLACK)
        game_surface = pygame.Surface((GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE))
        game_surface.fill(BLACK)
        
        # First, color the last 50 visited cells
        draw_visited_cells(game_surface, visited_positions)
        
        # Then draw the grid lines
        draw_grid(game_surface)
        draw_food(game_surface, game.food)
        draw_snake(game_surface, game.snake, tongue_visible)

        # Draw raycasting lines:
        draw_raycast_lines_to_wall(game_surface, game)
        draw_raycast_lines_to_body(game_surface, game)
        
        screen.blit(game_surface, (0,0))
        draw_info_panel(screen, game, state, last_reward, predicted_action)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
