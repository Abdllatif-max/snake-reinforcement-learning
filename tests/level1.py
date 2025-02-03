import pygame
import sys
import torch
import torch.nn as nn
import numpy as np
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

# For 22-input Snake: 
#   8 distances-to-body + 8 distances-to-wall + 4 food flags + 2 direction = 22
STATE_DIM = 22  
# 3 possible actions: turn left, go straight, turn right
ACTION_DIM = 3    

MODEL_CHECKPOINT = "best_dqn_snake.pth"  # Adjust if needed
UNDO_STACK_SIZE = 200  # Maximum states to remember for "undo"

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
# 2) SNAKE ENVIRONMENT (Same as training, 22 inputs)
############################################
class SnakeGame22:
    """
    Environment returns a 22-D state:
       - 8 distances to snake body in 8 directions
       - 8 distances to walls in those 8 directions
       - 4 binary food direction bits
       - 2 floats for (dx, dy) of the snake's heading

    Actions: 0=turn left, 1=straight, 2=turn right
    """
    # 8 directions (N, NE, E, SE, S, SW, W, NW):
    DIRECTIONS_8 = [(-1,0), (-1,1), (0,1), (1,1),
                    (1,0),  (1,-1), (0,-1), (-1,-1)]
    
    # Maps current direction => turn left => new direction
    LEFT_TURN = {
        (-1,0): (0,-1),  # up -> left
        (1,0):  (0,1),   # down -> right
        (0,-1): (1,0),   # left -> down
        (0,1):  (-1,0),  # right -> up
    }
    # Maps current direction => turn right => new direction
    RIGHT_TURN = {
        (-1,0): (0,1),   # up -> right
        (1,0):  (0,-1),  # down -> left
        (0,-1): (-1,0),  # left -> up
        (0,1):  (1,0),   # right -> down
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = (-1, 0)  # Start facing up
        self.food = self._place_food()
        self.done = False
        self.steps = 0
        self.cumulative_reward = 0.0
        return self._get_state()

    def _place_food(self):
        while True:
            fx = np.random.randint(0, GRID_SIZE)
            fy = np.random.randint(0, GRID_SIZE)
            if (fx, fy) not in self.snake:
                return (fx, fy)

    def _distance_to_wall(self, hx, hy, dx, dy):
        """
        Step outward until out of bounds to measure distance to wall.
        """
        dist = 0
        cx, cy = hx, hy
        while 0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE:
            cx += dx
            cy += dy
            dist += 1
        return dist

    def _distance_to_body(self, hx, hy, dx, dy):
        """
        Step outward until you hit a snake body part or go out of bounds.
        If never found, return the distance to out-of-bounds instead.
        """
        dist = 0
        cx, cy = hx, hy
        while True:
            cx += dx
            cy += dy
            dist += 1
            if not (0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE):
                return dist  # No body found, we just left grid
            if (cx, cy) in self.snake:
                return dist

    def _get_food_direction_bits(self, hx, hy):
        """
        4 bits: [food_up, food_down, food_left, food_right]
        """
        fx, fy = self.food
        up = 1 if fx < hx else 0
        down = 1 if fx > hx else 0
        left = 1 if fy < hy else 0
        right = 1 if fy > hy else 0
        return [up, down, left, right]

    def _get_state(self):
        """
        Return array of length 22:
         8 distances to body
         8 distances to wall
         4 bits food direction
         2 floats direction
        """
        hx, hy = self.snake[0]
        # 8 distances to body
        body_dists = []
        for (dx, dy) in self.DIRECTIONS_8:
            body_dists.append(float(self._distance_to_body(hx, hy, dx, dy)))
        # 8 distances to wall
        wall_dists = []
        for (dx, dy) in self.DIRECTIONS_8:
            wall_dists.append(float(self._distance_to_wall(hx, hy, dx, dy)))
        # 4 bits for food direction
        food_bits = self._get_food_direction_bits(hx, hy)
        # 2 for direction
        dx, dy = self.direction
        dir_vec = [float(dx), float(dy)]

        return np.array(body_dists + wall_dists + food_bits + dir_vec, dtype=np.float32)

    def step(self, action):
        """
        0=turn left, 1=straight, 2=turn right
        """
        if self.done:
            return self._get_state(), 0.0, True

        # Update direction
        if action == 0:
            self.direction = self.LEFT_TURN[self.direction]
        elif action == 2:
            self.direction = self.RIGHT_TURN[self.direction]
        # If action==1 => go straight, do nothing

        # Move
        hx, hy = self.snake[0]
        dx, dy = self.direction
        new_head = (hx + dx, hy + dy)

        # Collision check
        if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE):
            self.done = True
            reward = -10.0
            self.cumulative_reward += reward
            return self._get_state(), reward, True

        if new_head in self.snake:
            self.done = True
            reward = -15.0
            self.cumulative_reward += reward
            return self._get_state(), reward, True

        self.snake.insert(0, new_head)

        reward = -0.05  # small step penalty
        if new_head == self.food:
            reward += 15.0
            self.food = self._place_food()

        else:
            # remove tail
            self.snake.pop()

        self.steps += 1
        self.cumulative_reward += reward
        return self._get_state(), reward, False

    def snapshot(self):
        """
        Return a deep copy of environment state so we can revert later.
        We'll store enough to reconstruct exactly.
        """
        return {
            "snake": list(self.snake),
            "direction": self.direction,
            "food": self.food,
            "done": self.done,
            "steps": self.steps,
            "cumulative_reward": self.cumulative_reward
        }

    def restore(self, snap):
        """
        Restore environment from snapshot.
        """
        self.snake = list(snap["snake"])
        self.direction = snap["direction"]
        self.food = snap["food"]
        self.done = snap["done"]
        self.steps = snap["steps"]
        self.cumulative_reward = snap["cumulative_reward"]

############################################
# 3) LOAD PRETRAINED MODEL
############################################
device = torch.device("cpu")  # or "cuda" if GPU available
model = DQN(STATE_DIM, ACTION_DIM).to(device)
model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
model.eval()

############################################
# 4) PYGAME-BASED VISUALIZATION & DEBUG
############################################
def init_pygame():
    pygame.init()
    pygame.display.set_caption("Snake Debug Evaluator (22-input DQN)")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    return screen

def draw_grid(surface):
    # Vertical lines
    for x in range(GRID_SIZE):
        pygame.draw.line(surface, (60,60,60), (x*CELL_SIZE, 0), (x*CELL_SIZE, GRID_SIZE*CELL_SIZE))
    # Horizontal lines
    for y in range(GRID_SIZE):
        pygame.draw.line(surface, (60,60,60), (0, y*CELL_SIZE), (GRID_SIZE*CELL_SIZE, y*CELL_SIZE))

def draw_snake(surface, snake):
    # Draw snake segments
    for i, seg in enumerate(snake):
        rect = pygame.Rect(seg[1]*CELL_SIZE, seg[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        color = (0,200,0) if i == 0 else (0,255,0)
        pygame.draw.rect(surface, color, rect)

def draw_food(surface, food):
    rect = pygame.Rect(food[1]*CELL_SIZE, food[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, (255,0,0), rect)

def directions_8():
    # 8 directions (N, NE, E, SE, S, SW, W, NW)
    return [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]

def distance_ray_endpoint(game: SnakeGame22, dx, dy):
    """
    For drawing lines: find end of ray (either snake or wall).
    """
    hx, hy = game.snake[0]
    steps = 0
    cx, cy = hx, hy
    while True:
        cx += dx
        cy += dy
        steps += 1
        # out of bounds => revert to last valid
        if not (0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE):
            return (cx-dx, cy-dy)
        if (cx, cy) in game.snake:
            return (cx, cy)

def draw_raycast_lines(surface, game: SnakeGame22):
    """
    Draw lines from snake head in 8 directions until collision with wall or body.
    """
    head_x, head_y = game.snake[0]
    head_center = (head_y*CELL_SIZE + CELL_SIZE//2,
                   head_x*CELL_SIZE + CELL_SIZE//2)
    color_line = (0, 255, 255)
    for (dx, dy) in directions_8():
        end = distance_ray_endpoint(game, dx, dy)
        if end is not None:
            ex, ey = end
            end_center = (ey*CELL_SIZE + CELL_SIZE//2, ex*CELL_SIZE + CELL_SIZE//2)
            pygame.draw.line(surface, color_line, head_center, end_center, 2)

def draw_info_panel(surface, game: SnakeGame22, state, last_reward, predicted_action):
    """
    Right side panel: show:
      - steps, total reward
      - last step's reward
      - snake length
      - model's input states (22)
    """
    font = pygame.font.SysFont("Consolas", 18)
    # A background for the info panel
    panel_rect = pygame.Rect(GRID_SIZE*CELL_SIZE, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(surface, (30,30,30), panel_rect)

    # Lines
    y_offset = 10
    line_spacing = 22
    def blit_line(text, color=(255,255,255)):
        nonlocal y_offset
        txt_surf = font.render(text, True, color)
        surface.blit(txt_surf, (GRID_SIZE*CELL_SIZE+10, y_offset))
        y_offset += line_spacing

    blit_line(f"Steps: {game.steps}")
    blit_line(f"SnakeLen: {len(game.snake)}")
    blit_line(f"FoodEaten: {len(game.snake)-1}")  # or track separately
    blit_line(f"CumulR: {game.cumulative_reward:.2f}")
    blit_line(f"LastR: {last_reward:.2f}")
    action_names = {0:"Left",1:"Straight",2:"Right"}
    if predicted_action is not None:
        blit_line(f"Action: {action_names[predicted_action]}", (255,215,0))

    # Now the 22 inputs:
    # Indices: 0..7 => body dist, 8..15 => wall dist, 16..19 => food bits, 20..21 => direction
    blit_line("State vector (22):", (0,255,255))
    # Body Dist (8)
    body_dist = state[0:8]
    blit_line("  BodyDist(8):")
    for i in range(0, 8, 4):
        blit_line(f"    {body_dist[i:i+4]}")
    # Wall Dist (8)
    wall_dist = state[8:16]
    blit_line("  WallDist(8):")
    for i in range(0, 8, 4):
        blit_line(f"    {wall_dist[i:i+4]}")
    # Food bits (4)
    food_bits = state[16:20]
    blit_line("  FoodBits(4)")
    blit_line(f"  {food_bits.tolist()}")
    # Direction (2)
    direction_vals = state[20:22]
    blit_line(f"  Dir(2): {direction_vals.tolist()}")

############################################
# 5) MAIN LOOP
############################################
def main():
    screen = init_pygame()
    clock = pygame.time.Clock()

    game = SnakeGame22()
    state = game._get_state()
    last_reward = 0.0
    predicted_action = None

    # Keep snapshots to allow "undo" (left arrow)
    # We'll store (snapshot, last_reward) so we can revert
    history_stack = deque(maxlen=UNDO_STACK_SIZE)

    def push_history():
        snap = game.snapshot()
        history_stack.append((snap, state.copy(), last_reward))

    # Initial push of state 0
    push_history()

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RIGHT:
                    # Step forward with the model
                    if not game.done:
                        # Predict
                        s_tensor = torch.FloatTensor(state).unsqueeze(0)
                        with torch.no_grad():
                            q_values = model(s_tensor)
                        predicted_action = q_values.argmax(dim=1).item()

                        # Take snapshot before stepping
                        push_history()

                        # Step
                        next_state, r, done = game.step(predicted_action)
                        last_reward = r
                        state = next_state
                    else:
                        # If done, optionally reset or do nothing
                        pass

                elif event.key == pygame.K_LEFT:
                    # Undo: revert to previous snapshot if available
                    if len(history_stack) > 1:
                        # Pop current
                        history_stack.pop()
                        # Now top is the previous
                        snap, old_state, old_last_reward = history_stack[-1]
                        game.restore(snap)
                        state = old_state
                        last_reward = old_last_reward
                        predicted_action = None

        # Drawing
        screen.fill((0,0,0))

        # Game area
        game_surface = pygame.Surface((GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE))
        game_surface.fill((20,20,20))
        draw_grid(game_surface)
        draw_snake(game_surface, game.snake)
        draw_food(game_surface, game.food)
        draw_raycast_lines(game_surface, game)
        screen.blit(game_surface, (0,0))

        # Info panel
        draw_info_panel(screen, game, state, last_reward, predicted_action)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
