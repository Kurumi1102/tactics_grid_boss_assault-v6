# game_logic.py
import random
from units import Unit, Tank, Knight, AD, PLAYER_UNIT_SPECS
from boss import Boss
from agent import get_game_state_for_q_table 

class GameLogic:
    # ... (Phần __init__ và các hàm khác không đổi nhiều) ...
    def __init__(self, grid_size=4, max_rounds=9, agent_instance=None):
        self.grid_size = grid_size; self.max_rounds = max_rounds
        self.grid_units = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        self.boss = Boss(agent=agent_instance); self.current_round = 0
        self.player_max_accumulation = {name:spec["max_accumulation"] for name,spec in PLAYER_UNIT_SPECS.items()}
        self.player_current_accumulation = {}; self.units_placed_this_round_count = 0
        self.max_units_to_place_round_1 = 7; self.max_units_to_place_later_rounds = 2
        self.game_phase = "INITIALIZING"; self.action_log = []
        self.units_destroyed_this_round_by_boss = 0

    def start_new_game(self): # ... (Giữ nguyên)
        self.grid_units = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.boss.current_hp = self.boss.max_hp; self.boss.current_rage = 0
        for skill_key in self.boss.skills: self.boss.skills[skill_key]["cd_timer"] = 0
        self.current_round = 0; self.player_current_accumulation = self.player_max_accumulation.copy()
        self.action_log = ["Game Started (New Episode)."]; self._setup_new_round()
        return get_game_state_for_q_table(self) 
    def _regenerate_player_accumulation(self): # ... (Giữ nguyên)
        if self.current_round > 1:
            for unit_name in PLAYER_UNIT_SPECS:
                if self.player_current_accumulation[unit_name] < self.player_max_accumulation[unit_name]: self.player_current_accumulation[unit_name] += 1
            self.action_log.append(f"Player unit stock regenerated.")
    def _setup_new_round(self): # ... (Giữ nguyên)
        self.current_round += 1; self.units_placed_this_round_count = 0 
        self.boss.decrement_cooldowns(); self.units_destroyed_this_round_by_boss = 0
        if self.current_round > 1: self._regenerate_player_accumulation()
        self.action_log.append(f"--- Round {self.current_round} ---"); self.game_phase = "PLACEMENT"
    def get_max_units_to_place_this_round(self): return self.max_units_to_place_round_1 if self.current_round == 1 else self.max_units_to_place_later_rounds # ... (Giữ nguyên)
    def can_place_more_units_this_round(self): return self.units_placed_this_round_count < self.get_max_units_to_place_this_round() # ... (Giữ nguyên)
    def place_unit_from_stock(self, unit_name_to_place, r, c): # ... (Giữ nguyên)
        if self.game_phase != "PLACEMENT": return False, "Not in placement phase."
        if not self.can_place_more_units_this_round(): return False, f"Placement limit reached."
        if self.player_current_accumulation.get(unit_name_to_place, 0) <= 0: return False, f"No {unit_name_to_place}s left."
        if self.grid_units[r][c] is not None: return False, "Cell is occupied."
        unit_class = PLAYER_UNIT_SPECS[unit_name_to_place]["class"]
        unit_instance = unit_class(position=(r,c)); self.grid_units[r][c] = unit_instance
        self.player_current_accumulation[unit_name_to_place] -= 1; self.units_placed_this_round_count += 1
        self.action_log.append(f"Placed {unit_instance.name} at ({r},{c}). Stock: {self.player_current_accumulation[unit_name_to_place]}. Placed: {self.units_placed_this_round_count}.")
        return True, f"Placed {unit_instance.name}."
    def end_placement_phase(self): self.action_log.append("Placement phase ended."); return self.process_player_attack() # ... (Giữ nguyên)
    def process_player_attack(self): # ... (Giữ nguyên reward logic)
        self.game_phase = "PLAYER_ATTACK"; total_player_damage = 0; current_log = ["Player attacks:"]
        active_units = any(self.grid_units[r][c] for r in range(self.grid_size) for c in range(self.grid_size))
        if not active_units and self.current_round > 0 :
            current_log.append("No player units on board to attack."); self.action_log.extend(current_log); self.game_phase = "BOSS_ATTACK"
            next_state_dict = get_game_state_for_q_table(self); return "boss_turn", "No player units. Boss's turn.", 0, next_state_dict, 0, False
        for r_idx in range(self.grid_size):
            for c_idx in range(self.grid_size):
                unit = self.grid_units[r_idx][c_idx]
                if unit and unit.attack_power > 0: total_player_damage += unit.attack_power; current_log.append(f"- {unit.name} ({r_idx},{c_idx}) deals {unit.attack_power} damage.")
        boss_died = False
        if total_player_damage > 0:
            boss_died = self.boss.take_damage(total_player_damage); current_log.append(f"Boss takes {total_player_damage} total. HP: {self.boss.current_hp}")
        elif active_units: current_log.append("Player units dealt no damage.")
        self.action_log.extend(current_log); reward_for_boss = -(total_player_damage * 1.5)
        done = False; status_code = "boss_turn"; message = "Boss's turn."
        if boss_died:
            self.action_log.append("Boss Defeated!"); self.game_phase = "GAME_OVER"; reward_for_boss += -100; done = True
            status_code = "game_over_boss_defeated"; message = "Boss Defeated!"
        next_state_dict = get_game_state_for_q_table(self)
        if not done: self.game_phase = "BOSS_ATTACK"
        return status_code, message, total_player_damage, next_state_dict, reward_for_boss, done

    def process_boss_attack(self): 
        self.game_phase = "BOSS_ATTACK"; current_log = ["Boss's turn:"]; animation_triggers = []
        self.units_destroyed_this_round_by_boss = 0; reward_for_boss_action = 0

        current_state_dict_for_agent = get_game_state_for_q_table(self)
        chosen_skill_key, skill_params_val, action_idx = self.boss.choose_action_by_agent(current_state_dict_for_agent, self.grid_units)
        
        if action_idx is None and self.boss.agent is not None: pass

        if not chosen_skill_key: # ... (Giữ nguyên)
            current_log.append(self.boss.last_skill_message or "Boss does nothing."); self.action_log.extend(current_log)
            next_s_dict = get_game_state_for_q_table(self); done = self.check_game_over_conditions_for_done()
            current_r = 0;
            if done and self.boss.current_hp>0 and self.current_round>=self.max_rounds: current_r=150
            if not done: self.game_phase="ROUND_END"
            return "round_end", self.boss.last_skill_message or "Boss nothing.", [], next_s_dict, current_r, done, current_state_dict_for_agent, action_idx

        if not self.boss.apply_skill_effect_and_cd(chosen_skill_key): # ... (Giữ nguyên)
            current_log.append(self.boss.last_skill_message); self.action_log.extend(current_log)
            next_s_dict = get_game_state_for_q_table(self); done = self.check_game_over_conditions_for_done()
            current_r = -2;
            if done and self.boss.current_hp>0 and self.current_round>=self.max_rounds: current_r=150
            if not done: self.game_phase="ROUND_END"
            return "round_end", self.boss.last_skill_message, [], next_s_dict, current_r, done, current_state_dict_for_agent, action_idx

        current_log.append(self.boss.last_skill_message)
        skill_info = self.boss.skills[chosen_skill_key]
        damage_per_hit_instance = skill_info.get("damage", 0)
        is_unblockable = skill_info.get("unblockable", False)
        actual_hit_coords_for_animation = [] 
        
        if chosen_skill_key == "normal_attack":
            # skill_params_val is a list like [(r,c)] or empty
            if skill_params_val and skill_params_val[0][0] < self.grid_size and skill_params_val[0][1] < self.grid_size and \
               self.grid_units[skill_params_val[0][0]][skill_params_val[0][1]]:
                r,c = skill_params_val[0]; unit = self.grid_units[r][c]; unit_name = unit.name
                current_log.append(f"- Attacks {unit.name} at ({r},{c}) for {damage_per_hit_instance} damage.")
                if unit.take_damage(damage_per_hit_instance): 
                    reward_for_boss_action += self.get_kill_reward(unit_name)
                    self.grid_units[r][c] = None; self.units_destroyed_this_round_by_boss+=1
                    current_log.append(f"  - {unit.name} destroyed!")
                else: reward_for_boss_action += (damage_per_hit_instance * 1.5) 
                actual_hit_coords_for_animation.append((r,c))
            else: reward_for_boss_action -= 1 
            if actual_hit_coords_for_animation: animation_triggers.append({"type": "normal_attack", "targets": actual_hit_coords_for_animation})
        
        elif chosen_skill_key == "horizontal_shot" or chosen_skill_key == "vertical_shot":
            SHOT_INSTANCES = 4 
            damage_instances_left = SHOT_INSTANCES
            
            line_idx = skill_params_val.get("line_idx", 0) # Default to 0 if not found
            direction = skill_params_val.get("direction", None)
            
            line_coords_ordered = []
            anim_type = chosen_skill_key

            if chosen_skill_key == "horizontal_shot":
                current_log.append(f"- Bắn Ngang (4 charges) on row {line_idx} dir {direction}:")
                if direction == "rtl": # Right to Left
                    for c_idx in range(self.grid_size - 1, -1, -1): line_coords_ordered.append((line_idx, c_idx))
                else: # Default Left to Right (ltr)
                    for c_idx in range(self.grid_size): line_coords_ordered.append((line_idx, c_idx))
            else: # vertical_shot
                current_log.append(f"- Bắn Dọc (4 charges) on column {line_idx} dir {direction}:")
                if direction == "btt": # Bottom to Top
                    for r_idx in range(self.grid_size - 1, -1, -1): line_coords_ordered.append((r_idx, line_idx))
                else: # Default Top to Bottom (ttb)
                    for r_idx in range(self.grid_size): line_coords_ordered.append((r_idx, line_idx))

            hit_at_least_one_target_in_line = False
            for r, c in line_coords_ordered:
                if damage_instances_left <= 0: break 
                unit_in_cell = self.grid_units[r][c]
                if unit_in_cell:
                    hit_at_least_one_target_in_line = True
                    actual_hit_coords_for_animation.append((r,c)) 
                    if isinstance(unit_in_cell, Tank) and not is_unblockable:
                        current_log.append(f"  - Beam reaches Tank {unit_in_cell.name} at ({r},{c}). HP: {unit_in_cell.current_hp}")
                        hits_on_tank = 0
                        while damage_instances_left > 0 and unit_in_cell.current_hp > 0:
                            current_log.append(f"    - Tank takes 1 hit from charge. ({damage_instances_left-1} charges left)")
                            unit_in_cell.take_damage(damage_per_hit_instance)
                            reward_for_boss_action += (damage_per_hit_instance * 1.5) 
                            damage_instances_left -= 1; hits_on_tank += 1
                            if unit_in_cell.current_hp <= 0:
                                current_log.append(f"    - Tank {unit_in_cell.name} destroyed after {hits_on_tank} hits!")
                                reward_for_boss_action += self.get_kill_reward(unit_in_cell.name) 
                                self.grid_units[r][c] = None; self.units_destroyed_this_round_by_boss += 1
                                break 
                        if unit_in_cell and unit_in_cell.current_hp > 0: 
                            current_log.append(f"    - Tank {unit_in_cell.name} survives. Skill exhausted on Tank.")
                            damage_instances_left = 0                             
                    else: 
                        current_log.append(f"  - Beam hits {unit_in_cell.name} at ({r},{c}) for 1 charge.")
                        unit_name_hit = unit_in_cell.name
                        if unit_in_cell.take_damage(damage_per_hit_instance):
                            reward_for_boss_action += self.get_kill_reward(unit_name_hit)
                            self.grid_units[r][c] = None; self.units_destroyed_this_round_by_boss += 1
                            current_log.append(f"    - {unit_name_hit} destroyed!")
                        else: 
                            reward_for_boss_action += (damage_per_hit_instance * 1.5)
                            current_log.append(f"    - {unit_name_hit} survives. Beam continues...")
                        damage_instances_left -=1 
            if not hit_at_least_one_target_in_line: reward_for_boss_action -= 2 
            if actual_hit_coords_for_animation: 
                # For sequential animation, sort targets based on direction if not already
                # The line_coords_ordered IS ALREADY in the correct animation sequence.
                # We just need to ensure actual_hit_coords_for_animation preserves this or is re-sorted.
                # However, actual_hit_coords_for_animation is built in order, so it should be fine.
                animation_triggers.append({"type": anim_type, "targets": actual_hit_coords_for_animation})


        elif chosen_skill_key == "ultimate": # skill_params_val is a list here
            current_log.append(f"- Ultimate strikes {len(skill_params_val)} locations:"); targets_hit_ulti = []
            unique_params = list(set(skill_params_val)) 
            units_hit_by_ulti_count = 0
            for r_target, c_target in unique_params:
                unit = self.grid_units[r_target][c_target]; targets_hit_ulti.append((r_target,c_target))
                if unit:
                    units_hit_by_ulti_count +=1; unit_name_hit = unit.name
                    current_log.append(f"  - Hits ({r_target},{c_target}) by {unit.name} for {damage_per_hit_instance} damage.")
                    if unit.take_damage(damage_per_hit_instance): 
                        reward_for_boss_action += self.get_kill_reward(unit_name_hit)
                        self.grid_units[r_target][c_target]=None; self.units_destroyed_this_round_by_boss+=1
                        current_log.append(f"    - {unit.name} destroyed!")
                    else: reward_for_boss_action += (damage_per_hit_instance * 1.5)
                else: current_log.append(f"  - Hits empty ({r_target},{c_target}).")
            if units_hit_by_ulti_count == 0 and len(unique_params)>0: reward_for_boss_action -= 3 
            elif units_hit_by_ulti_count > 2: reward_for_boss_action += units_hit_by_ulti_count 
            if targets_hit_ulti: animation_triggers.append({"type": "ultimate_hit", "targets": targets_hit_ulti})
        
        elif chosen_skill_key == "heal":
            if self.boss.current_hp < self.boss.max_hp * 0.3 : reward_for_boss_action += 8 
            elif self.boss.current_hp < self.boss.max_hp * 0.6: reward_for_boss_action += 4 
            elif self.boss.current_hp > self.boss.max_hp * 0.9: reward_for_boss_action -= 3 
            else: reward_for_boss_action += 0.5 
            animation_triggers.append({"type": "boss_heal"})
        
        self.action_log.extend(current_log)
        
        next_state_dict_for_agent = get_game_state_for_q_table(self)
        done = False; status_ui = "round_end"; msg_ui = "Boss turn finished."

        player_units_left = any(self.grid_units[r][c] for r in range(self.grid_size) for c in range(self.grid_size))
        if not player_units_left and self.units_destroyed_this_round_by_boss > 0:
            self.action_log.append("All player units destroyed by Boss this round!")
            reward_for_boss_action += 150 
            done = True; self.game_phase = "GAME_OVER"
            status_ui = "game_over_player_wiped"; msg_ui = "All player units destroyed!"
        
        if not done:
            is_game_over_by_round_limit_flag, _ = self.check_game_over_conditions() 
            is_boss_dead_final = self.boss.current_hp <= 0
            if is_boss_dead_final: 
                reward_for_boss_action += -100; done = True
            elif is_game_over_by_round_limit_flag and self.boss.current_hp > 0: 
                reward_for_boss_action += 150; done = True 
            elif self.boss.current_hp > 0: reward_for_boss_action += 1 

        if not done: self.game_phase = "ROUND_END"
        return status_ui, msg_ui, animation_triggers, next_state_dict_for_agent, reward_for_boss_action, done, current_state_dict_for_agent, action_idx

    # ... (get_kill_reward, check_game_over_conditions_for_done, check_game_over_conditions, 
    #      proceed_to_next_round, get_action_log giữ nguyên) ...
    def get_kill_reward(self, unit_name):
        if unit_name == "AD": return 7
        if unit_name == "Knight": return 4
        if unit_name == "Tank": return 3
        return 0
    def check_game_over_conditions_for_done(self):
        if self.current_round >= self.max_rounds and self.boss.current_hp > 0: return True
        if self.boss.current_hp <= 0: return True
        return False
    def check_game_over_conditions(self):
        if self.current_round >= self.max_rounds and self.boss.current_hp > 0:
            self.game_phase = "GAME_OVER"; return True, f"Game Over: Boss survives {self.max_rounds} rounds! Player loses."
        if self.boss.current_hp <= 0:
            self.game_phase = "GAME_OVER"; return True, "Game Over: Boss defeated! Player wins!"
        return False, "Game continues."
    def proceed_to_next_round(self):
        is_over, msg = self.check_game_over_conditions()
        current_state_dict = get_game_state_for_q_table(self)
        if is_over: self.action_log.append(msg); return "game_over", msg, current_state_dict
        if self.current_round >= self.max_rounds and self.boss.current_hp > 0:
            self.game_phase = "GAME_OVER"; final_msg = f"Game Over: Boss survived {self.max_rounds} rounds!"
            self.action_log.append(final_msg); return "game_over", final_msg, current_state_dict
        self._setup_new_round()
        return "new_round_placement", f"Starting Round {self.current_round}. Place units.", get_game_state_for_q_table(self)
    def get_action_log(self, tail=0):
        if tail > 0 and len(self.action_log) > tail: return self.action_log[-tail:]
        return self.action_log