# agent.py
import numpy as np
import random
import pickle 

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

class QLearningTableAgent:
    def __init__(self, learning_rate=0.0001, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.999995, 
                 min_exploration_rate=0.005,
                 boss_skills_ref=None):
        self.lr = learning_rate; self.gamma = discount_factor
        self.epsilon = exploration_rate; self.epsilon_decay = exploration_decay; self.epsilon_min = min_exploration_rate
        self.boss_skills_ref = boss_skills_ref
        q_table_dims = ( 
            HP_BINS, RAGE_BINS,
            CD_STATES_PER_SKILL["horizontal_shot"], CD_STATES_PER_SKILL["vertical_shot"], CD_STATES_PER_SKILL["heal"],
            NUM_TANK_BINS, NUM_KNIGHT_BINS, NUM_AD_BINS,
            ROUND_BINS,
            NUM_ACTIONS
        )
        try:
            self.q_table = np.zeros(q_table_dims)
            # print(f"Q-Table initialized with dimensions: {q_table_dims}, Total size: {self.q_table.size}")
        except Exception as e:
            print(f"ERROR initializing Q-Table: {e}"); raise

    def _discretize_state(self, game_state_dict):
        hp_val = game_state_dict["boss_hp"]
        max_hp = game_state_dict.get("boss_max_hp", 50)
        if max_hp == 0: hp_idx = 0
        else: hp_idx = min(int((hp_val / max_hp) * HP_BINS), HP_BINS - 1)
        if hp_val >= max_hp * ((HP_BINS - 0.5) / HP_BINS if HP_BINS > 0 else 1.0) : hp_idx = HP_BINS - 1 # Handle edge case for full HP
        if hp_val <= 0: hp_idx = 0
            
        rage_idx = game_state_dict["boss_rage"]
        cd_hshot_idx = min(game_state_dict["skill_cooldowns"]["horizontal_shot"], CD_STATES_PER_SKILL["horizontal_shot"] - 1)
        cd_vshot_idx = min(game_state_dict["skill_cooldowns"]["vertical_shot"], CD_STATES_PER_SKILL["vertical_shot"] - 1)
        cd_heal_idx = min(game_state_dict["skill_cooldowns"]["heal"], CD_STATES_PER_SKILL["heal"] - 1)
        num_tanks = game_state_dict["unit_counts"]["Tank"]
        num_knights = game_state_dict["unit_counts"]["Knight"]
        num_ads = game_state_dict["unit_counts"]["AD"]
        tank_idx = min(num_tanks, NUM_TANK_BINS - 1)
        knight_idx = min(num_knights, NUM_KNIGHT_BINS - 1)
        ad_idx = min(num_ads, NUM_AD_BINS - 1)
        round_idx = min(game_state_dict["current_round"] - 1, ROUND_BINS - 1); round_idx = max(0, round_idx)
        return (hp_idx, rage_idx, cd_hshot_idx, cd_vshot_idx, cd_heal_idx, tank_idx, knight_idx, ad_idx, round_idx)

    def choose_action(self, discrete_state_tuple, available_skill_keys, grid_units_for_targeting):
        available_action_indices = [idx for idx, sk_key in ACTION_MAP_AGENT.items() if sk_key in available_skill_keys]
        if not available_action_indices: return None, [], None 
        action_idx = -1
        if np.random.rand() <= self.epsilon:
            action_idx = random.choice(available_action_indices)
        else:
            q_values_for_state = self.q_table[discrete_state_tuple]
            valid_q_values = {idx: q_values_for_state[idx] for idx in available_action_indices}
            if not valid_q_values: action_idx = random.choice(available_action_indices)
            else: action_idx = max(valid_q_values, key=valid_q_values.get)
        chosen_skill_key = ACTION_MAP_AGENT.get(action_idx)
        if not chosen_skill_key: return None, [], None
        # skill_params will now be a dictionary for H/V shots: {"line_idx": idx, "direction": "rtl" or "ltr" or "ttb" or "btt"}
        skill_params = self._get_heuristic_skill_params(chosen_skill_key, grid_units_for_targeting)
        return chosen_skill_key, skill_params, action_idx

    def _get_heuristic_skill_params(self, skill_key, grid_units):
        player_unit_positions=[]; ads_positions=[]; knights_positions=[]; tanks_positions=[]
        grid_h = len(grid_units); grid_w = len(grid_units[0]) if grid_h > 0 else 0
        for r_loop in range(grid_h): # Renamed r to r_loop
            for c_loop in range(grid_w): # Renamed c to c_loop
                unit = grid_units[r_loop][c_loop]
                if unit:
                    player_unit_positions.append((r_loop,c_loop))
                    if unit.name=="AD": ads_positions.append((r_loop,c_loop))
                    elif unit.name=="Knight": knights_positions.append((r_loop,c_loop))
                    elif unit.name=="Tank": tanks_positions.append((r_loop,c_loop))
        
        params = {} # Use a dictionary for H/V shots, list for others
        if skill_key=="normal_attack":
            target_list_normal = [] # Keep as list for normal attack
            if ads_positions: target_list_normal = [random.choice(ads_positions)]
            elif knights_positions: target_list_normal = [random.choice(knights_positions)]
            elif tanks_positions: target_list_normal = [random.choice(tanks_positions)]
            elif player_unit_positions: target_list_normal = [random.choice(player_unit_positions)]
            return target_list_normal # Return list for normal attack

        elif skill_key=="horizontal_shot":
            best_row,max_targets=-1,-1
            for r_idx in range(grid_h):
                count=sum(1 for c_idx in range(grid_w) if grid_units[r_idx][c_idx] and grid_units[r_idx][c_idx].name in ["AD","Knight"])
                if count>max_targets:max_targets=count;best_row=r_idx
            
            params["line_idx"] = best_row if best_row != -1 else (random.randint(0,grid_h-1) if grid_h > 0 else 0)
            params["direction"] = random.choice(["ltr", "rtl"]) # left-to-right or right-to-left
        
        elif skill_key=="vertical_shot":
            best_col,max_targets=-1,-1
            for c_idx in range(grid_w):
                count=sum(1 for r_idx in range(grid_h) if grid_units[r_idx][c_idx] and grid_units[r_idx][c_idx].name in ["AD","Knight"])
                if count>max_targets:max_targets=count;best_col=c_idx

            params["line_idx"] = best_col if best_col != -1 else (random.randint(0,grid_w-1) if grid_w > 0 else 0)
            params["direction"] = random.choice(["ttb", "btt"]) # top-to-bottom or bottom-to-top

        elif skill_key=="ultimate":
            target_list_ulti = [] # Keep as list for ultimate
            targets_ulti_temp = ads_positions+knights_positions+tanks_positions; random.shuffle(targets_ulti_temp)
            if len(targets_ulti_temp)<6:
                empty_cells=[(r_loop,c_loop) for r_loop in range(grid_h) for c_loop in range(grid_w) if grid_units[r_loop][c_loop] is None]
                random.shuffle(empty_cells); targets_ulti_temp.extend(empty_cells[:6-len(targets_ulti_temp)])
            target_list_ulti = targets_ulti_temp[:6]
            return target_list_ulti # Return list for ultimate
        
        # For heal, params can be an empty dict or None, game_logic doesn't use it for heal
        return params


    def learn(self, current_discrete_state, action_idx, reward, next_discrete_state, done):
        if action_idx is None or action_idx < 0 or action_idx >= NUM_ACTIONS: return
        old_value = self.q_table[current_discrete_state][action_idx]
        next_max_q = 0 if done else np.max(self.q_table[next_discrete_state])
        new_value = old_value + self.lr * (reward + self.gamma * next_max_q - old_value)
        self.q_table[current_discrete_state][action_idx] = new_value
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

    def load(self, filepath="q_table_boss.pkl"):
        try:
            with open(filepath, 'rb') as f: self.q_table = pickle.load(f)
            # print(f"Q-Table loaded from {filepath}")
        except FileNotFoundError: print(f"No Q-Table at {filepath}.")
        except Exception as e: print(f"Error loading Q-Table: {e}")

    def save(self, filepath="q_table_boss.pkl"):
        try:
            with open(filepath, 'wb') as f: pickle.dump(self.q_table, f)
            # print(f"Q-Table saved to {filepath}")
        except Exception as e: print(f"Error saving Q-Table: {e}")

def get_game_state_for_q_table(game_logic_instance):
    boss=game_logic_instance.boss; grid=game_logic_instance.grid_units
    state_dict={}; state_dict["boss_hp"]=boss.current_hp; state_dict["boss_max_hp"]=boss.max_hp
    state_dict["boss_rage"]=boss.current_rage
    state_dict["skill_cooldowns"]={
        "horizontal_shot":boss.skills["horizontal_shot"]["cd_timer"],
        "vertical_shot":boss.skills["vertical_shot"]["cd_timer"],
        "heal":boss.skills["heal"]["cd_timer"],
    }
    unit_counts={"Tank":0,"Knight":0,"AD":0}
    for r_loop in range(game_logic_instance.grid_size): # Renamed r
        for c_loop in range(game_logic_instance.grid_size): # Renamed c
            unit=grid[r_loop][c_loop]
            if unit and unit.name in unit_counts: unit_counts[unit.name]+=1
    state_dict["unit_counts"]=unit_counts
    state_dict["current_round"]=game_logic_instance.current_round
    return state_dict