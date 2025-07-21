# boss.py
import random

class Boss:
    def __init__(self, agent=None): 
        self.max_hp = 60
        self.current_hp = self.max_hp
        self.max_rage = 3; self.current_rage = 0
        self.skills = { 
            "normal_attack": {"cd":0,"cd_timer":0,"rage_gain":1,"damage":1,"name":"Đánh Thường"},
            "horizontal_shot": {"cd":2,"cd_timer":0,"rage_gain":1,"damage":1,"name":"Bắn Ngang"},
            "vertical_shot": {"cd":2,"cd_timer":0,"rage_gain":1,"damage":1,"name":"Bắn Dọc"},
            "heal": {"cd":3,"cd_timer":0,"rage_gain":0,"heal_amount":10,"name":"Hồi Máu"},
            "ultimate": {"cd":0,"cd_timer":0,"rage_cost":3,"damage":2,"name":"Ultimate","unblockable":True}
        }
        self.last_skill_message = ""; self.agent = agent 

    def take_damage(self, amount): # ... (Giữ nguyên)
        self.current_hp -= amount
        if self.current_hp < 0: self.current_hp = 0
        return self.current_hp <= 0
    def gain_rage(self, amount): self.current_rage = min(self.current_rage + amount, self.max_rage) # ... (Giữ nguyên)
    def use_rage(self, amount): # ... (Giữ nguyên)
        if self.current_rage >= amount: self.current_rage -= amount; return True
        return False
    def apply_skill_effect_and_cd(self, skill_key): # ... (Giữ nguyên)
        skill = self.skills[skill_key]
        if skill_key == "heal": self.current_hp = min(self.current_hp + skill["heal_amount"], self.max_hp)
        elif skill_key == "ultimate":
            if not self.use_rage(skill["rage_cost"]): return False
        else: self.gain_rage(skill["rage_gain"])
        if skill_key not in ["normal_attack", "ultimate"]: skill["cd_timer"] = skill["cd"]
        return True
    def decrement_cooldowns(self): # ... (Giữ nguyên)
        for key in ["horizontal_shot", "vertical_shot", "heal"]:
            if self.skills[key]["cd_timer"] > 0: self.skills[key]["cd_timer"] -= 1
    def get_available_skills_keys(self): # ... (Giữ nguyên)
        available = []
        fixed_skill_order = ["normal_attack", "horizontal_shot", "vertical_shot", "heal", "ultimate"]
        for key in fixed_skill_order:
            skill_data = self.skills[key]
            if skill_data["cd_timer"] == 0:
                if key == "ultimate":
                    if self.current_rage >= skill_data["rage_cost"]: available.append(key)
                else: available.append(key)
        return available

    def choose_action_by_agent(self, current_game_state_dict_for_q_table, grid_units_for_targeting):
        if self.agent:
            available_keys = self.get_available_skills_keys()
            if not available_keys:
                self.last_skill_message = "Boss has no available skills (QAgent)."
                return None, {}, None # skill, params_dict, action_idx

            discrete_state_tuple = self.agent._discretize_state(current_game_state_dict_for_q_table)
            chosen_skill_key, skill_params_dict, action_idx = self.agent.choose_action(
                discrete_state_tuple, available_keys, grid_units_for_targeting
            )

            if chosen_skill_key:
                skill_name_display = self.skills[chosen_skill_key]['name']
                param_str = ""
                if chosen_skill_key == "normal_attack" and skill_params_dict: # skill_params_dict is a list here
                     if skill_params_dict : param_str = f"on ({skill_params_dict[0][0]},{skill_params_dict[0][1]})"
                elif chosen_skill_key in ["horizontal_shot", "vertical_shot"] and skill_params_dict: # skill_params_dict is a dict
                     param_str = f"on line {skill_params_dict.get('line_idx')} dir: {skill_params_dict.get('direction')}"
                
                self.last_skill_message = f"Boss (QAgent) uses {skill_name_display} {param_str}."
                if chosen_skill_key == "ultimate": self.last_skill_message = f"Boss (QAgent) uses {skill_name_display}!" # Params not in message for ulti
                if chosen_skill_key == "heal": self.last_skill_message = f"Boss (QAgent) uses {skill_name_display}."
            else:
                self.last_skill_message = "Boss (QAgent) could not decide."
            return chosen_skill_key, skill_params_dict, action_idx
        else:
            chosen_skill_key, skill_params_dict = self.fallback_choose_action_ai(grid_units_for_targeting)
            return chosen_skill_key, skill_params_dict, None 

    def fallback_choose_action_ai(self, grid_units):
        available_skills = self.get_available_skills_keys()
        if not available_skills:
            self.last_skill_message = "Boss has no available skills (fallback)."
            return None, {} # Return empty dict for params

        player_unit_positions = [(r, c) for r in range(4) for c in range(4) if grid_units[r][c] is not None]
        chosen_skill_key = ""
        # ... (rest of fallback AI logic to choose chosen_skill_key) ...
        if "ultimate" in available_skills and len(player_unit_positions) >= 3: chosen_skill_key = "ultimate"
        elif "heal" in available_skills and self.current_hp <= self.max_hp * 0.4 and self.current_hp < self.max_hp: chosen_skill_key = "heal"
        else:
            damage_skills = [s for s in available_skills if s not in ["heal", "ultimate"] and player_unit_positions]
            if not damage_skills and "normal_attack" in available_skills and player_unit_positions: damage_skills.append("normal_attack")
            damage_skills = list(set(damage_skills))
            if damage_skills: chosen_skill_key = random.choice(damage_skills)
            elif "heal" in available_skills and self.current_hp < self.max_hp: chosen_skill_key = "heal"
            elif "normal_attack" in available_skills and player_unit_positions: chosen_skill_key = "normal_attack"
            elif available_skills: chosen_skill_key = random.choice(available_skills)
            else: self.last_skill_message = "Boss AI (fallback) could not decide."; return None, {}


        skill_params_val = {} # Use dict for H/V shots, list for others
        skill_name_display = self.skills[chosen_skill_key]['name']

        if chosen_skill_key == "normal_attack":
            temp_list_params = []
            if player_unit_positions:
                target_pos = random.choice(player_unit_positions); temp_list_params = [target_pos]
                self.last_skill_message = f"Boss (fallback) uses {skill_name_display} on ({target_pos[0]},{target_pos[1]})."
            else: self.last_skill_message = f"Boss (fallback) tries {skill_name_display}, no targets."; return "normal_attack", {} 
            skill_params_val = temp_list_params # Assign list
        elif chosen_skill_key == "horizontal_shot":
            skill_params_val["line_idx"] = random.randint(0, 3)
            skill_params_val["direction"] = random.choice(["ltr", "rtl"])
            self.last_skill_message = f"Boss (fallback) uses {skill_name_display} on row {skill_params_val['line_idx']} ({skill_params_val['direction']})."
        elif chosen_skill_key == "vertical_shot":
            skill_params_val["line_idx"] = random.randint(0, 3)
            skill_params_val["direction"] = random.choice(["ttb", "btt"])
            self.last_skill_message = f"Boss (fallback) uses {skill_name_display} on column {skill_params_val['line_idx']} ({skill_params_val['direction']})."
        elif chosen_skill_key == "ultimate":
            temp_list_params_ulti = []
            possible_targets = player_unit_positions[:] 
            empty_cells = [(r,c) for r in range(4) for c in range(4) if grid_units[r][c] is None]
            random.shuffle(possible_targets); random.shuffle(empty_cells)
            temp_list_params_ulti = (possible_targets + empty_cells)[:6]
            self.last_skill_message = f"Boss (fallback) uses {skill_name_display}!"
            skill_params_val = temp_list_params_ulti # Assign list
        elif chosen_skill_key == "heal":  
            self.last_skill_message = f"Boss (fallback) uses {skill_name_display}."
            skill_params_val = {} # Empty dict for heal
        
        if chosen_skill_key not in ["heal", "ultimate"] and not player_unit_positions and chosen_skill_key in ["normal_attack", "horizontal_shot", "vertical_shot"] :
             # Check if chosen skill is target-dependent and there are no targets
             if chosen_skill_key == "normal_attack" and not skill_params_val: # Normal attack specifically needs a target for its param list
                 self.last_skill_message = f"Boss (fallback) chose {skill_name_display} but board empty."
                 return chosen_skill_key, {}
             elif chosen_skill_key in ["horizontal_shot", "vertical_shot"]: # H/V shots can still be fired on empty lines
                 pass # Allow firing on empty lines

        return chosen_skill_key, skill_params_val