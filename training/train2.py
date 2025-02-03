import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from IPython.display import clear_output

############################################
# 1) Configurable Parameters
############################################
EPISODES = 2000
GRID_SIZE = 15
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.002
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.01

OLD_CHECKPOINT_PATH = "best_dqn_snake.pth"      # The old model to resume from
SAVE_BEST_PATH      = "best_dqn_snake_level2.pth"  # New file for saving best model

############################################
# 2) SnakeGame Environment (Level 2)
############################################
class SnakeGame:
    """
    Same environment, but we add an extra reward/penalty
    for turning vs. going straight:
      +1 for action=1 (straight)
      -1 for action=0 or 2 (turn left or right)
    """
    def __init__(self):
        # 8 directional offsets for N, NE, E, SE, S, SW, W, NW
        self.directions_8 = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                             (1, 0),  (1, -1), (0, -1), (-1, -1)]

        # For turning left/right relative to current direction
        self.left_turn = {
            (-1,0): (0,-1),  # up -> left
            (1,0):  (0,1),   # down -> right
            (0,-1): (1,0),   # left -> down
            (0,1):  (-1,0),  # right -> up
        }
        self.right_turn = {
            (-1,0): (0,1),   # up -> right
            (1,0):  (0,-1),  # down -> left
            (0,-1): (-1,0),  # left -> up
            (0,1):  (1,0),   # right -> down
        }

        self.reset()

    def reset(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = (-1, 0)  # Start facing upward
        self.food = self._place_food()
        self.done = False
        self.steps = 0
        return self._get_state()

    def _place_food(self):
        while True:
            fx = np.random.randint(0, GRID_SIZE)
            fy = np.random.randint(0, GRID_SIZE)
            if (fx, fy) not in self.snake:
                return (fx, fy)

    def _distance_to_wall(self, head_x, head_y, dx, dy):
        distance = 0
        cur_x, cur_y = head_x, head_y
        while 0 <= cur_x < GRID_SIZE and 0 <= cur_y < GRID_SIZE:
            cur_x += dx
            cur_y += dy
            distance += 1
        return distance

    def _distance_to_body(self, head_x, head_y, dx, dy):
        distance = 0
        cur_x, cur_y = head_x, head_y
        while True:
            cur_x += dx
            cur_y += dy
            distance += 1
            # out of bounds => no collision with body
            if not (0 <= cur_x < GRID_SIZE and 0 <= cur_y < GRID_SIZE):
                return distance
            if (cur_x, cur_y) in self.snake:
                return distance

    def _get_food_direction(self, head_x, head_y):
        fx, fy = self.food
        up    = 1 if fx < head_x else 0
        down  = 1 if fx > head_x else 0
        left  = 1 if fy < head_y else 0
        right = 1 if fy > head_y else 0
        return [up, down, left, right]

    def _get_state(self):
        """
        Returns a 22-D state:
          8 distances to snake body,
          8 distances to wall,
          4 binary food direction bits,
          2 floats for (dx, dy).
        """
        head_x, head_y = self.snake[0]

        # Distances to body
        body_dists = []
        for (dx, dy) in self.directions_8:
            body_dists.append(float(self._distance_to_body(head_x, head_y, dx, dy)))

        # Distances to wall
        wall_dists = []
        for (dx, dy) in self.directions_8:
            wall_dists.append(float(self._distance_to_wall(head_x, head_y, dx, dy)))

        # Food direction
        food_dir = self._get_food_direction(head_x, head_y)

        # Current heading
        dx, dy = self.direction
        direction_vec = [float(dx), float(dy)]

        return np.array(body_dists + wall_dists + food_dir + direction_vec, dtype=np.float32)

    def step(self, action):
        """
        3 actions: 0=left, 1=straight, 2=right
        Additional 'body style' reward:
         +1 if action=1 (straight),
         -1 if action=0 or 2 (turn).
        """
        if self.done:
            return self._get_state(), 0.0, True

        # Turn left or right if needed
        if action == 0:  # left
            self.direction = self.left_turn[self.direction]
        elif action == 2:  # right
            self.direction = self.right_turn[self.direction]
        # action=1 => straight => do nothing

        # Body-style reward
        if action == 1:
            turn_reward = +0.1
        else:
            turn_reward = -1.0

        # Move snake head
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # Collision check
        if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE or
            new_head in self.snake):
            self.done = True
            return self._get_state(), -10.0 + turn_reward, True

        self.snake.insert(0, new_head)
        self.steps += 1

        # Check if food is eaten
        if new_head == self.food:
            reward = 15.0 + turn_reward
            self.food = self._place_food()
        else:
            self.snake.pop()
            # small negative step cost
            reward = -0.05 + turn_reward

        return self._get_state(), reward, False

############################################
# 3) DQN Network (22 -> 3)
############################################
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # 3 hidden layers of 256 + a final 128 layer
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
# 4) DQN Agent (Soft-Target, Replay Buffer)
############################################
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.epsilon = INITIAL_EPSILON
        self.memory = deque(maxlen=50000)

        self.q_network = DQN(state_dim, action_dim)
        self.target_q_network = DQN(state_dim, action_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        self.q_network.eval()  # BatchNorm uses running stats in eval mode
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()

    def store_experience(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        self.q_network.train()  # BN in training mode
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states_t      = torch.FloatTensor(states)
        actions_t     = torch.LongTensor(actions).unsqueeze(1)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q-values
        q_values = self.q_network(states_t).gather(1, actions_t)

        # Next action from online net
        next_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)

        # Target Q from target net
        with torch.no_grad():
            target_q_next = self.target_q_network(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1 - dones_t) * GAMMA * target_q_next

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self):
        # Polyak averaging for target network
        for target_param, local_param in zip(self.target_q_network.parameters(),
                                             self.q_network.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1 - TAU) * target_param.data)

############################################
# 5) Resume Training Loop (Level 2)
############################################
if __name__ == "__main__":
    STATE_DIM  = 22  # 8 body + 8 wall + 4 food direction + 2 heading
    ACTION_DIM = 3   # left, straight, right

    # Create environment & agent
    env   = SnakeGame()
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)

    # 1) Load old model weights => start from previously learned policy
    old_weights = torch.load(OLD_CHECKPOINT_PATH, map_location="cpu")
    agent.q_network.load_state_dict(old_weights)
    agent.target_q_network.load_state_dict(agent.q_network.state_dict())
    print(f"Loaded old model from: {OLD_CHECKPOINT_PATH}")

    episode_rewards = []
    episode_steps = []
    best_reward = -np.inf  # Will track best reward at Level 2

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            # Action selection
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            # Store and train
            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()

            # Update for next step
            state = next_state
            total_reward += reward
            steps += 1

        # Decay exploration
        agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)

        # Soft update target network
        agent.soft_update()

        # Track stats
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        # Save new best model if improved
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.q_network.state_dict(), SAVE_BEST_PATH)
            print(f"🔥 New Best Model at Episode {episode+1} | Reward: {total_reward:.2f}")

        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_steps  = np.mean(episode_steps[-50:])
            print(f"Episode {episode+1} | "
                  f"Avg Reward (last 50): {avg_reward:.2f} | "
                  f"Avg Steps: {avg_steps:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print("\n🎉 Level 2 Training Complete!")
    print(f"🏆 Best Reward Achieved: {best_reward:.2f}")
    print(f"💾 Saved Model: '{SAVE_BEST_PATH}'")

    ############################################
    # 6) Plot Results
    ############################################
    plt.figure(figsize=(12, 5))

    # Plot Rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label="Episode Reward (Level 2)")
    plt.title("Level 2: Encouraging Straighter Moves")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    # Plot Steps
    plt.subplot(1, 2, 2)
    plt.plot(episode_steps, label="Episode Steps", color="orange")
    plt.title("Steps Survived Per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"📊 Average Reward (Last 50): {np.mean(episode_rewards[-50:]):.2f}")
    print(f"📈 Average Steps (Last 50): {np.mean(episode_steps[-50:]):.1f}")
