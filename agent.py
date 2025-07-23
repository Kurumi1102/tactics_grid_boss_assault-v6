# agent.py
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque # For replay buffer

# --- State Discretization Parameters ---
HP_BINS = 5
RAGE_BINS = 4
CD_STATES_PER_SKILL = {"horizontal_shot": 3, "vertical_shot": 3, "heal": 4}
NUM_TANK_BINS = 4
NUM_KNIGHT_BINS = 4
NUM_AD_BINS = 5
ROUND_BINS = 9

ACTION_MAP_AGENT = {
    0: "normal_attack", 1: "horizontal_shot", 2: "vertical_shot", 3: "heal", 4: "ultimate"
}
NUM_ACTIONS = len(ACTION_MAP_AGENT)

# Define the Neural Network for the Q-value approximation
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # ONE hidden layer with ReLU activation, followed by output layer
        self.fc1 = nn.Linear(input_dim, 128) # First layer: input to hidden
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim) # Second (output) layer: hidden to output (Q-values)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        return self.fc2(x)

# New Agent Class using DQN
class DQNAgent:
    def __init__(self, learning_rate=0.0005, discount_factor=0.95, # Adjusted LR
                 exploration_rate=1.0, exploration_decay=0.99999, # Adjusted decay
                 min_exploration_rate=0.005,
                 boss_skills_ref=None,
                 replay_buffer_size=50000, batch_size=64, target_update_freq=100):
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        self.boss_skills_ref = boss_skills_ref

        # Define input and output dimensions for the neural network
        self.input_dim = 9 # (hp, rage, cd_hshot, cd_vshot, cd_heal, tanks, knights, ads, round)
        self.output_dim = NUM_ACTIONS

        # Policy Network (main Q-network)
        self.policy_net = DQN(self.input_dim, self.output_dim)
        # Target Network (for stable Q-value calculation)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Set target network to evaluation mode (no gradients, no dropout)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Mean Squared Error Loss for Q-value prediction

        # Replay Buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0 # Counter for target network updates

        # Device configuration (CPU or GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def _discretize_state(self, game_state_dict):
        """
        Converts a raw game state dictionary into a normalized numpy array (float32)
        suitable for neural network input.
        """
        hp_val = game_state_dict["boss_hp"]
        max_hp = game_state_dict.get("boss_max_hp", 50) # Default to 50 if not found for safety

        # Normalize HP (0-1 range)
        hp_normalized = hp_val / max_hp if max_hp > 0 else 0.0

        # Rage (already 0-3, normalize to 0-1)
        rage_normalized = game_state_dict["boss_rage"] / (RAGE_BINS - 1)

        # Cooldowns (already capped, normalize to 0-1)
        cd_hshot_normalized = game_state_dict["skill_cooldowns"]["horizontal_shot"] / (CD_STATES_PER_SKILL["horizontal_shot"] - 1)
        cd_vshot_normalized = game_state_dict["skill_cooldowns"]["vertical_shot"] / (CD_STATES_PER_SKILL["vertical_shot"] - 1)
        cd_heal_normalized = game_state_dict["skill_cooldowns"]["heal"] / (CD_STATES_PER_SKILL["heal"] - 1)
        
        # Unit counts (already capped, normalize to 0-1)
        tank_normalized = game_state_dict["unit_counts"]["Tank"] / (NUM_TANK_BINS - 1)
        knight_normalized = game_state_dict["unit_counts"]["Knight"] / (NUM_KNIGHT_BINS - 1)
        ad_normalized = game_state_dict["unit_counts"]["AD"] / (NUM_AD_BINS - 1)
        
        # Round index (already capped and min-clamped, normalize to 0-1)
        round_normalized = max(0, game_state_dict["current_round"] - 1) / (ROUND_BINS - 1)
        
        # Create a numpy array from normalized values
        state_vector = np.array([
            hp_normalized,
            rage_normalized,
            cd_hshot_normalized,
            cd_vshot_normalized,
            cd_heal_normalized,
            tank_normalized,
            knight_normalized,
            ad_normalized,
            round_normalized
        ], dtype=np.float32)

        return state_vector

    def choose_action(self, state_vector, available_skill_keys, grid_units_for_targeting):
        """
        Chooses an action based on epsilon-greedy strategy.
        """
        available_action_indices = [idx for idx, sk_key in ACTION_MAP_AGENT.items() if sk_key in available_skill_keys]
        if not available_action_indices:
            return None, [], None # No available actions

        action_idx = -1
        if np.random.rand() <= self.epsilon:
            # Exploration: choose a random available action
            action_idx = random.choice(available_action_indices)
        else:
            # Exploitation: choose action with highest Q-value from the policy network
            state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).to(self.device)
            self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).squeeze(0)
            self.policy_net.train()

            masked_q_values = q_values.clone()
            for i in range(self.output_dim):
                if i not in available_action_indices:
                    masked_q_values[i] = -float('inf')

            action_idx = torch.argmax(masked_q_values).item()

            if action_idx not in available_action_indices:
                 action_idx = random.choice(available_action_indices)

        chosen_skill_key = ACTION_MAP_AGENT.get(action_idx)
        if not chosen_skill_key:
            return None, [], None
        
        skill_params = self._get_heuristic_skill_params(chosen_skill_key, grid_units_for_targeting)
        return chosen_skill_key, skill_params, action_idx

    def _get_heuristic_skill_params(self, skill_key, grid_units):
        """
        Heuristic to determine skill parameters (targets, directions) based on the chosen skill.
        This part is identical to the original QLearningTableAgent's logic.
        """
        player_unit_positions=[]; ads_positions=[]; knights_positions=[]; tanks_positions=[]
        grid_h = len(grid_units); grid_w = len(grid_units[0]) if grid_h > 0 else 0
        for r_loop in range(grid_h):
            for c_loop in range(grid_w):
                unit = grid_units[r_loop][c_loop]
                if unit:
                    player_unit_positions.append((r_loop,c_loop))
                    if unit.name=="AD": ads_positions.append((r_loop,c_loop))
                    elif unit.name=="Knight": knights_positions.append((r_loop,c_loop))
                    elif unit.name=="Tank": tanks_positions.append((r_loop,c_loop))
        
        params = {}
        if skill_key=="normal_attack":
            target_list_normal = []
            if ads_positions: target_list_normal = [random.choice(ads_positions)]
            elif knights_positions: target_list_normal = [random.choice(knights_positions)]
            elif tanks_positions: target_list_normal = [random.choice(tanks_positions)]
            elif player_unit_positions: target_list_normal = [random.choice(player_unit_positions)]
            return target_list_normal

        elif skill_key=="horizontal_shot":
            best_row,max_targets=-1,-1
            for r_idx in range(grid_h):
                count=sum(1 for c_idx in range(grid_w) if grid_units[r_idx][c_idx] and grid_units[r_idx][c_idx].name in ["AD","Knight"])
                if count>max_targets:max_targets=count;best_row=r_idx
            
            params["line_idx"] = best_row if best_row != -1 else (random.randint(0,grid_h-1) if grid_h > 0 else 0)
            params["direction"] = random.choice(["ltr", "rtl"])
        
        elif skill_key=="vertical_shot":
            best_col,max_targets=-1,-1
            for c_idx in range(grid_w):
                count=sum(1 for r_idx in range(grid_h) if grid_units[r_idx][c_idx] and grid_units[r_idx][c_idx].name in ["AD","Knight"])
                if count>max_targets:max_targets=count;best_col=c_idx

            params["line_idx"] = best_col if best_col != -1 else (random.randint(0,grid_w-1) if grid_w > 0 else 0)
            params["direction"] = random.choice(["ttb", "btt"])

        elif skill_key=="ultimate":
            target_list_ulti = []
            targets_ulti_temp = ads_positions+knights_positions+tanks_positions; random.shuffle(targets_ulti_temp)
            if len(targets_ulti_temp)<6:
                empty_cells=[(r_loop,c_loop) for r_loop in range(grid_h) for c_loop in range(grid_w) if grid_units[r_loop][c_loop] is None]
                random.shuffle(empty_cells); targets_ulti_temp.extend(empty_cells[:6-len(targets_ulti_temp)])
            target_list_ulti = targets_ulti_temp[:6]
            return target_list_ulti
        
        return params

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience tuple (S, A, R, S', Done) in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self, current_state_vector, action_idx, reward, next_state_vector, done):
        """
        Performs a single learning step on the DQN.
        Samples a batch, computes loss, and updates network weights.
        """
        if action_idx is None or action_idx < 0 or action_idx >= NUM_ACTIONS:
            return

        # Add current experience to replay buffer
        self.remember(current_state_vector, action_idx, reward, next_state_vector, done)

        # Only start learning if enough experiences are in the buffer for a batch
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        # Unpack the batch into separate tensors
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert unpacked lists to PyTorch tensors and move to device
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(list(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(list(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(list(dones), dtype=torch.bool).to(self.device)

        # Calculate current Q values (Q(s, a)) using the policy network
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calculate target Q values (r + gamma * max(Q(s', a')))
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Compute the loss
        loss = self.criterion(current_q_values, target_q_values)

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1) # Gradient clipping
        self.optimizer.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filepath="dqn_boss_agent.pth"):
        """Loads the DQN policy and target network states, optimizer state, and agent parameters."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.update_count = checkpoint['update_count']
            
            self.policy_net.to(self.device)
            self.target_net.to(self.device)
            self.target_net.eval()
            print(f"DQN Agent loaded from {filepath}. Epsilon: {self.epsilon:.4f}")
        except FileNotFoundError:
            print(f"No DQN Agent model found at {filepath}. Starting training from scratch or playing with untrained agent.")
        except Exception as e:
            print(f"Error loading DQN Agent: {e}")

    def save(self, filepath="dqn_boss_agent.pth"):
        """Saves the DQN policy and target network states, optimizer state, and agent parameters."""
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'update_count': self.update_count,
            }, filepath)
            print(f"DQN Agent saved to {filepath}")
        except Exception as e:
            print(f"Error saving DQN Agent: {e}")

def get_game_state_for_q_table(game_logic_instance):
    boss=game_logic_instance.boss
    grid=game_logic_instance.grid_units
    
    state_dict={}
    state_dict["boss_hp"]=boss.current_hp
    state_dict["boss_max_hp"]=boss.max_hp
    state_dict["boss_rage"]=boss.current_rage
    state_dict["skill_cooldowns"]={
        "horizontal_shot":boss.skills["horizontal_shot"]["cd_timer"],
        "vertical_shot":boss.skills["vertical_shot"]["cd_timer"],
        "heal":boss.skills["heal"]["cd_timer"],
    }
    
    unit_counts={"Tank":0,"Knight":0,"AD":0}
    for r_loop in range(game_logic_instance.grid_size):
        for c_loop in range(game_logic_instance.grid_size):
            unit=grid[r_loop][c_loop]
            if unit and unit.name in unit_counts:
                unit_counts[unit.name]+=1
    state_dict["unit_counts"]=unit_counts
    
    state_dict["current_round"]=game_logic_instance.current_round
    return state_dict