# main.py
import sys
import os
import time
import random
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout,
                             QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
                             QMessageBox, QFrame, QTextEdit,
                             QSizePolicy, QSpacerItem)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QFont, QMovie, QPalette, QBrush, QPixmap

from units import PLAYER_UNIT_SPECS
from game_logic import GameLogic
from agent import QLearningTableAgent, get_game_state_for_q_table

# --- RL Agent Configuration ---
TRAIN_MODE = False
NUM_EPISODES_TO_TRAIN = 200000
SAVE_AGENT_EVERY_N_EPISODES = 5000
AGENT_MODEL_FILE = "q_table_boss_agent.pkl"
LOG_STATS_EVERY_N_EPISODES = 500
TRAINING_STATS_FILE = "training_stats.csv" # Tên file CSV để lưu thống kê

class TacticsGridWindow(QMainWindow):
    # ... (Phần __init__ và init_ui không thay đổi) ...
    def __init__(self, agent_to_use=None):
        super().__init__()
        self.setWindowTitle("Tactics Grid – Boss Assault (Q-Table RL)")
        self.setGeometry(100, 100, 900, 850)
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.forest_bg_path = os.path.join(base_path, "assets", "forest_bg.jpg").replace("\\", "/")
        self.boss_gif_path = os.path.join(base_path, "assets", "boss_idle.gif").replace("\\", "/")
        self.game = GameLogic(agent_instance=agent_to_use)
        if agent_to_use and hasattr(self.game.boss, 'skills') and not agent_to_use.boss_skills_ref :
            agent_to_use.boss_skills_ref = self.game.boss.skills
        self.grid_buttons = [[None for _ in range(self.game.grid_size)] for _ in range(self.game.grid_size)]
        self.selected_unit_type_for_placement = None
        self.player_stock_buttons = {}
        self.cell_end_of_round_effects = {}
        self.boss_display_effect = {"type": None, "timer": None, "original_style": ""}
        self.short_term_animation_timers = []
        self.is_fast_mode_training = False
        self.current_episode_count = 0
        self.init_ui()

    def init_ui(self):
        palette = QPalette(); pixmap = QPixmap(self.forest_bg_path)
        if not pixmap.isNull():
            palette.setBrush(QPalette.Window, QBrush(pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)))
            self.setPalette(palette); self.setAutoFillBackground(True)
        else: self.setStyleSheet("QMainWindow { background-color: #2c3e50; }")
        self.central_widget = QWidget(); self.central_widget.setAutoFillBackground(False)
        self.central_widget.setStyleSheet("background: transparent;"); self.setCentralWidget(self.central_widget)
        overall_horizontal_layout = QHBoxLayout(self.central_widget); overall_horizontal_layout.addStretch(1)
        self.main_content_container = QWidget()
        self.main_content_container.setStyleSheet("background-color: rgba(20,20,30,0.75); border-radius:10px; padding:15px;")
        self.main_content_container.setFixedWidth(700)
        self.main_content_v_layout = QVBoxLayout(self.main_content_container)
        overall_horizontal_layout.addWidget(self.main_content_container); overall_horizontal_layout.addStretch(1)
        self.episode_label = QLabel("Episode: 0")
        self.episode_label.setFont(QFont("Arial",10,QFont.Bold))
        self.episode_label.setStyleSheet("color: #FFD700; background:transparent;"); self.episode_label.setAlignment(Qt.AlignRight)
        if not TRAIN_MODE: self.episode_label.hide()
        self.main_content_v_layout.addWidget(self.episode_label)
        self.round_label = QLabel(); self.round_label.setFont(QFont("Arial",16,QFont.Bold)); self.round_label.setAlignment(Qt.AlignCenter)
        self.round_label.setStyleSheet("color:white; background:transparent;"); self.main_content_v_layout.addWidget(self.round_label)
        stock_area_widget = QWidget(); stock_area_layout = QVBoxLayout(stock_area_widget); stock_area_layout.setContentsMargins(0,0,0,0)
        self.player_stock_info_label = QLabel("Player Unit Stock (Click to select, then click grid):")
        self.player_stock_info_label.setFont(QFont("Arial",11,QFont.Bold)); self.player_stock_info_label.setStyleSheet("color:#E0E0E0; background:transparent; margin-bottom:5px;")
        self.player_stock_info_label.setAlignment(Qt.AlignCenter); stock_area_layout.addWidget(self.player_stock_info_label)
        self.player_stock_buttons_layout = QHBoxLayout()
        for unit_name, spec in PLAYER_UNIT_SPECS.items():
            button = QPushButton(f"{unit_name}\n({spec['abbr']}) 0/0"); button.setFont(QFont("Arial",10)); button.setMinimumHeight(60); button.setCheckable(True)
            button.clicked.connect(lambda chk, name=unit_name: self.on_stock_unit_selected(name))
            self.player_stock_buttons[unit_name] = button; self.player_stock_buttons_layout.addWidget(button)
        stock_area_layout.addLayout(self.player_stock_buttons_layout); self.main_content_v_layout.addWidget(stock_area_widget)
        boss_display_container = QWidget(); boss_display_container.setStyleSheet("background:transparent;")
        boss_display_layout = QVBoxLayout(boss_display_container); boss_display_layout.setContentsMargins(0,5,0,5); boss_display_layout.setSpacing(5)
        self.boss_gif_label = QLabel(); self.boss_gif_label.setAlignment(Qt.AlignCenter); boss_gif_size = 160
        self.boss_gif_label.setFixedSize(boss_gif_size,boss_gif_size); self.boss_display_effect["original_style"] = "background-color:rgba(50,50,70,0.3); border:1px solid #444; border-radius:8px;"
        self.boss_gif_label.setStyleSheet(self.boss_display_effect["original_style"])
        if os.path.exists(self.boss_gif_path):
            self.boss_movie = QMovie(self.boss_gif_path)
            if self.boss_movie.isValid(): self.boss_movie.setScaledSize(QSize(boss_gif_size-10,boss_gif_size-10)); self.boss_gif_label.setMovie(self.boss_movie); self.boss_movie.start()
            else: self.boss_gif_label.setText("Boss GIF\nInvalid")
        else: self.boss_gif_label.setText("Boss GIF\nNot Found")
        boss_display_layout.addWidget(self.boss_gif_label,alignment=Qt.AlignCenter)
        self.boss_hp_label = QLabel(); self.boss_hp_label.setFont(QFont("Arial",12,QFont.Bold)); self.boss_hp_label.setAlignment(Qt.AlignCenter)
        self.boss_hp_label.setStyleSheet("color:white; background:transparent; padding:3px;"); boss_display_layout.addWidget(self.boss_hp_label,alignment=Qt.AlignCenter)
        self.main_content_v_layout.addWidget(boss_display_container,alignment=Qt.AlignCenter)
        self.action_log_display = QTextEdit(); self.action_log_display.setFont(QFont("Arial",10)); self.action_log_display.setReadOnly(True)
        self.action_log_display.setFixedHeight(100); self.action_log_display.setStyleSheet("background-color:rgba(10,10,20,0.7); color:#E0E0E0; border-radius:5px; padding:5px; border:1px solid #333;")
        self.main_content_v_layout.addWidget(self.action_log_display)
        self.grid_layout = QGridLayout(); self.grid_layout.setSpacing(5)
        grid_frame = QFrame(); grid_frame.setStyleSheet("background:transparent;"); grid_frame.setLayout(self.grid_layout)
        for r in range(self.game.grid_size):
            for c in range(self.game.grid_size):
                button = QPushButton(""); button.setFixedSize(80,80); button.setFont(QFont("Arial",10))
                button.clicked.connect(lambda checked, r_val=r, c_val=c: self.on_grid_cell_clicked(r_val,c_val))
                self.grid_buttons[r][c] = button; self.grid_layout.addWidget(button,r,c)
        self.main_content_v_layout.addWidget(grid_frame,alignment=Qt.AlignCenter)
        self.placement_info_label = QLabel(f"Can place 0 more units this round.")
        self.placement_info_label.setFont(QFont("Arial",11,QFont.Bold)); self.placement_info_label.setStyleSheet("color:#DDD; background:transparent; margin-top:5px;")
        self.placement_info_label.setAlignment(Qt.AlignCenter); self.main_content_v_layout.addWidget(self.placement_info_label)
        self.control_layout = QHBoxLayout()
        self.end_placement_button = QPushButton("End Placement & Attack")
        self.end_placement_button.setFont(QFont("Arial",12,QFont.Bold)); self.end_placement_button.setMinimumHeight(40)
        self.end_placement_button.setStyleSheet("QPushButton { background-color:#007BFF; color:white; border-radius:5px; padding:5px; } QPushButton:hover { background-color:#0056b3; } QPushButton:disabled { background-color:#555; color:#aaa; }")
        self.end_placement_button.clicked.connect(self.on_end_placement_clicked); self.control_layout.addWidget(self.end_placement_button)
        self.main_content_v_layout.addLayout(self.control_layout)

    def resizeEvent(self, event):
        super().resizeEvent(event); palette = QPalette(); pixmap = QPixmap(self.forest_bg_path)
        if not pixmap.isNull(): palette.setBrush(QPalette.Window, QBrush(pixmap.scaled(self.size(),Qt.KeepAspectRatioByExpanding,Qt.SmoothTransformation))); self.setPalette(palette)

    def start_new_game_ui(self, initial_state_dict_for_agent=None):
        self.reset_all_round_animations()
        if initial_state_dict_for_agent is None : self.game.start_new_game()
        self.selected_unit_type_for_placement = None
        self.set_controls_for_phase(self.game.game_phase)
        self.update_all_ui_displays()
        if self.current_episode_count == 0 or not TRAIN_MODE: self.log_message("New Game Started. Select units from stock to place.")
        return get_game_state_for_q_table(self.game)

    def update_stock_buttons_display(self):
        can_place_more=self.game.can_place_more_units_this_round()
        for unit_name,button in self.player_stock_buttons.items():
            spec=PLAYER_UNIT_SPECS[unit_name];current_stock=self.game.player_current_accumulation.get(unit_name,0);max_stock=self.game.player_max_accumulation.get(unit_name,0)
            button.setText(f"{unit_name} ({spec['abbr']})\n{current_stock}/{max_stock}")
            button_enabled=current_stock>0 and can_place_more and self.game.game_phase=="PLACEMENT"
            button.setEnabled(button_enabled)
            if self.selected_unit_type_for_placement==unit_name: button.setStyleSheet("QPushButton { background-color:lightgreen; color:black; border:2px solid green; padding:5px;} QPushButton:disabled { background-color:#444; color:#888; }")
            else: button.setStyleSheet("QPushButton { background-color:#555; border:1px solid #777; padding:5px; color:white;} QPushButton:hover { background-color:#666; } QPushButton:disabled { background-color:#444; color:#888; }")

    def on_stock_unit_selected(self,unit_name):
        if self.is_fast_mode_training and self.game.game_phase=="PLACEMENT":
            if self.game.can_place_more_units_this_round() and self.game.player_current_accumulation.get(unit_name,0)>0:
                empty_cells=[(r,c_grid) for r in range(self.game.grid_size) for c_grid in range(self.game.grid_size) if self.game.grid_units[r][c_grid] is None]
                if empty_cells: r_place,c_place=random.choice(empty_cells); self.game.place_unit_from_stock(unit_name,r_place,c_place); self.update_all_ui_displays()
            return
        if not self.game.can_place_more_units_this_round() and self.selected_unit_type_for_placement!=unit_name:
            self.log_message("Placement limit reached.",is_error=True)
            if self.selected_unit_type_for_placement is not None: pass
            else: self.selected_unit_type_for_placement=None
            self.update_stock_buttons_display(); return
        if self.game.player_current_accumulation.get(unit_name,0)<=0 and self.selected_unit_type_for_placement!=unit_name:
            self.log_message(f"No {unit_name}s left.",is_error=True); self.update_stock_buttons_display(); return
        if self.selected_unit_type_for_placement==unit_name: self.selected_unit_type_for_placement=None; self.log_message(f"Deselected {unit_name}.")
        else:
            if not self.game.can_place_more_units_this_round(): self.log_message("Placement limit. Cannot select new.",is_error=True)
            elif self.game.player_current_accumulation.get(unit_name,0)<=0: self.log_message(f"No {unit_name}s left to select.",is_error=True)
            else: self.selected_unit_type_for_placement=unit_name; self.log_message(f"Selected {unit_name}. Click grid.")
        self.update_stock_buttons_display()

    def reset_all_round_animations(self):
        for t in self.short_term_animation_timers: t.stop(); t.deleteLater();
        self.short_term_animation_timers.clear()
        for k,v in list(self.cell_end_of_round_effects.items()):
            if v.get("timer"): v["timer"].stop(); v["timer"].deleteLater()
        self.cell_end_of_round_effects.clear()
        if self.boss_display_effect.get("timer"): self.boss_display_effect["timer"].stop();self.boss_display_effect["timer"].deleteLater()
        if hasattr(self,'boss_gif_label') and self.boss_gif_label: self.boss_gif_label.setStyleSheet(self.boss_display_effect["original_style"])
        self.boss_display_effect["type"]=None; self.boss_display_effect["timer"]=None

    def get_cell_font_weight_style(self,r,c): return "font-weight:bold; color:black;" if self.game.grid_units[r][c] else "font-weight:normal; color:black;"

    def animate_boss_display_glow(self,color="rgba(144,238,144,0.6)"):
        if self.boss_display_effect.get("timer"): self.boss_display_effect["timer"].stop();self.boss_display_effect["timer"]=None
        style=f"background-color:{color}; border:2px solid lightgreen; border-radius:8px;"
        if hasattr(self,'boss_gif_label') and self.boss_gif_label: self.boss_gif_label.setStyleSheet(style)
        self.boss_display_effect["type"]="persistent_glow"

    def animate_boss_display_persistent_flash(self,color1="rgba(255,0,0,0.5)",color2="rgba(255,127,127,0.5)",interval=300):
        if hasattr(self,'boss_gif_label') and self.boss_gif_label:
            if self.boss_display_effect.get("type")=="persistent_glow": self.boss_gif_label.setStyleSheet(self.boss_display_effect["original_style"])
            if self.boss_display_effect.get("timer"): self.boss_display_effect["timer"].stop()
            flash_state={"is_color1":True}
            def flash():
                if not self.boss_gif_label:return
                bg_color=color1 if flash_state["is_color1"] else color2
                style=f"background-color:{bg_color}; border:2px solid red; border-radius:8px;"
                self.boss_gif_label.setStyleSheet(style); flash_state["is_color1"]=not flash_state["is_color1"]
            timer=QTimer(self);timer.timeout.connect(flash);timer.start(interval)
            self.boss_display_effect["type"]="persistent_flash";self.boss_display_effect["timer"]=timer;flash()

    def start_cell_persistent_flash(self,r,c,color1="yellow",color2="lightcoral",interval=350):
        if(r,c) in self.cell_end_of_round_effects and self.cell_end_of_round_effects[(r,c)].get("timer"):self.cell_end_of_round_effects[(r,c)]["timer"].stop()
        button=self.grid_buttons[r][c];font_style=self.get_cell_font_weight_style(r,c);flash_state={"is_color1":True}
        def flash():
            if not button:return
            bg_color=color1 if flash_state["is_color1"] else color2
            button.setStyleSheet(f"background-color:{bg_color}; {font_style} border-radius:5px;");flash_state["is_color1"]=not flash_state["is_color1"]
        timer=QTimer(self);timer.timeout.connect(flash);timer.start(interval)
        self.cell_end_of_round_effects[(r,c)]={"type":"persistent_flash","timer":timer,"data":{}};flash()

    def animate_sequential_shot(self,targets,highlight_color="orange",persist_color="darkorange",sequential_delay=300,highlight_duration=250):
        if not targets:return
        for i,(r_coord,c_coord) in enumerate(targets):
            button=self.grid_buttons[r_coord][c_coord];font_style=self.get_cell_font_weight_style(r_coord,c_coord)
            def light_up_cell(b=button,r_cell=r_coord,c_cell=c_coord,f_style=font_style):
                if not b:return
                b.setStyleSheet(f"background-color:{highlight_color}; {f_style} border-radius:5px;")
                persist_timer=QTimer(self);persist_timer.setSingleShot(True)
                persist_timer.timeout.connect(lambda:self.set_cell_persistent_style(r_cell,c_cell,persist_color,f_style))
                persist_timer.start(highlight_duration);self.short_term_animation_timers.append(persist_timer)
            effect_timer=QTimer(self);effect_timer.setSingleShot(True);effect_timer.timeout.connect(light_up_cell)
            effect_timer.start(i*sequential_delay);self.short_term_animation_timers.append(effect_timer)

    def set_cell_persistent_style(self,r,c,color,font_style):
        button=self.grid_buttons[r][c]
        if not button:return
        button.setStyleSheet(f"background-color:{color}; {font_style} border-radius:5px;")
        self.cell_end_of_round_effects[(r,c)]={"type":"persistent_style","timer":None,"data":{"color":color}}

    def animate_tank_block(self,r,c,color1="white",color2="dodgerblue",interval=166,flashes=3):
        button=self.grid_buttons[r][c];original_unit_color="slategray";font_style=self.get_cell_font_weight_style(r,c)
        if not button:return
        flash_state={"is_color1":True,"count":0,"max_flashes":flashes*2};timer=QTimer(self)
        def flash():
            if not button:timer.stop();return
            flash_state["count"]+=1
            if flash_state["count"]>flash_state["max_flashes"]:
                button.setStyleSheet(f"background-color:{original_unit_color}; {font_style} border-radius:5px;");timer.stop()
                if timer in self.short_term_animation_timers:self.short_term_animation_timers.remove(timer)
                timer.deleteLater();return
            bg_color=color1 if flash_state["is_color1"] else color2
            button.setStyleSheet(f"background-color:{bg_color}; {font_style} border-radius:5px;");flash_state["is_color1"]=not flash_state["is_color1"]
        timer.timeout.connect(flash);timer.start(interval);self.short_term_animation_timers.append(timer);flash()

    def update_all_ui_displays(self):
        self.update_info_display(); self.update_stock_buttons_display()
        self.update_grid_display(); self.update_action_log_display()
        self.update_placement_info_label()
        if TRAIN_MODE and hasattr(self, 'episode_label'):
             self.episode_label.setText(f"Episode: {self.current_episode_count}/{NUM_EPISODES_TO_TRAIN} | R: {self.game.current_round}")
             if self.is_fast_mode_training: QApplication.processEvents() # Chỉ processEvents khi đang train nhanh

    def update_info_display(self):
        self.round_label.setText(f"Round: {self.game.current_round}/{self.game.max_rounds}")
        boss_text = f"Boss HP: {self.game.boss.current_hp}/{self.game.boss.max_hp} | Rage: {self.game.boss.current_rage}"
        self.boss_hp_label.setText(boss_text)
        if self.boss_display_effect["type"] is None and hasattr(self,'boss_gif_label') and self.boss_gif_label:
             self.boss_gif_label.setStyleSheet(self.boss_display_effect["original_style"])

    def update_placement_info_label(self):
        if self.game.game_phase == "PLACEMENT":
            remaining = self.game.get_max_units_to_place_this_round() - self.game.units_placed_this_round_count
            self.placement_info_label.setText(f"Can place {remaining} more units this round.")
            self.placement_info_label.show()
        else: self.placement_info_label.hide()

    def update_grid_display(self):
        for r_idx in range(self.game.grid_size):
            for c_idx in range(self.game.grid_size):
                unit=self.game.grid_units[r_idx][c_idx]; button=self.grid_buttons[r_idx][c_idx]
                font_style=self.get_cell_font_weight_style(r_idx,c_idx)
                if unit: button.setText(unit.get_display_text())
                else: button.setText("")
                if(r_idx,c_idx) in self.cell_end_of_round_effects: pass
                else:
                    if unit:
                        color="gray"
                        if unit.name=="Tank": color="slategray"
                        elif unit.name=="Knight": color="lightblue"
                        elif unit.name=="AD": color="lightcoral"
                        button.setStyleSheet(f"background-color:{color}; {font_style} border-radius:5px;")
                    else: button.setStyleSheet(f"background-color:rgba(80,80,80,0.7); {font_style} border-radius:5px;")

    def update_action_log_display(self):
        self.action_log_display.clear()
        log_text = "\n".join(self.game.get_action_log(tail=5 if self.is_fast_mode_training else 15))
        self.action_log_display.setText(log_text)
        self.action_log_display.verticalScrollBar().setValue(self.action_log_display.verticalScrollBar().maximum())

    def log_message(self, message, is_error=False, is_success=False):
        if not self.is_fast_mode_training:
            self.game.action_log.append(message)
        # Bỏ điều kiện log key stats vào UI, vì chúng sẽ được print ra console
        # elif "Episode" in message or "GAME OVER" in message or "saved" in message or "Avg Reward" in message or "Win Rate" in message:
        #     self.game.action_log.append(message)
        self.update_action_log_display()

    def on_grid_cell_clicked(self, r, c):
        if self.is_fast_mode_training and self.game.game_phase == "PLACEMENT": return
        if self.game.game_phase != "PLACEMENT": self.log_message("Not in placement phase.", is_error=True); return
        if self.selected_unit_type_for_placement is None: self.log_message("Select unit type from stock.", is_error=True); return
        unit_to_place = self.selected_unit_type_for_placement
        success, message = self.game.place_unit_from_stock(unit_to_place, r, c)
        self.log_message(message, is_error=not success, is_success=success)
        if success: self.selected_unit_type_for_placement = None; self.update_all_ui_displays()
        else: self.update_stock_buttons_display(); self.update_placement_info_label()

    def set_controls_for_phase(self, phase):
        is_placement = (phase == "PLACEMENT")
        self.end_placement_button.setEnabled(is_placement)
        for unit_name, button in self.player_stock_buttons.items():
            button_enabled = (is_placement and self.game.player_current_accumulation.get(unit_name,0)>0 and self.game.can_place_more_units_this_round())
            button.setEnabled(button_enabled)
        self.placement_info_label.setVisible(is_placement)
        for r_idx in range(self.game.grid_size):
            for c_idx in range(self.game.grid_size):
                if self.grid_buttons[r_idx][c_idx]: self.grid_buttons[r_idx][c_idx].setEnabled(is_placement)

    def on_end_placement_clicked(self):
        if self.is_fast_mode_training and self.game.game_phase == "PLACEMENT": return
        if self.game.game_phase != "PLACEMENT": return
        self.log_message("Player ends turn actions...")
        self.selected_unit_type_for_placement = None
        results = self.game.end_placement_phase()
        status_code, message, damage_to_boss = results[0], results[1], results[2]
        self.log_message(message)
        if damage_to_boss > 0 and status_code != "game_over_boss_defeated": self.animate_boss_display_persistent_flash()
        self.set_controls_for_phase(self.game.game_phase); self.update_all_ui_displays()
        if status_code == "game_over_boss_defeated": self.handle_game_over(message); return
        QTimer.singleShot(0 if self.is_fast_mode_training else 1000, self.execute_boss_turn)

    def execute_player_turn_for_training(self):
        self.is_fast_mode_training = True; self.game.game_phase = "PLACEMENT"
        num_to_place = self.game.get_max_units_to_place_this_round(); placed_count = 0
        available_types = [utype for utype, count in self.game.player_current_accumulation.items() if count > 0]
        random.shuffle(available_types)
        for _ in range(num_to_place):
            if not self.game.can_place_more_units_this_round() or not available_types: break
            unit_type = random.choice(available_types)
            if self.game.player_current_accumulation[unit_type] > 0:
                empty_cells = [(r,c) for r in range(self.game.grid_size) for c in range(self.game.grid_size) if self.game.grid_units[r][c] is None]
                if empty_cells:
                    r_place,c_place = random.choice(empty_cells)
                    success,_ = self.game.place_unit_from_stock(unit_type,r_place,c_place)
                    if success:
                        placed_count+=1
                        if self.game.player_current_accumulation[unit_type]==0:
                            if unit_type in available_types: available_types.remove(unit_type)
            if not available_types and placed_count < num_to_place:
                available_types = [utype for utype,count in self.game.player_current_accumulation.items() if count>0]
                if not available_types: break
                else: random.shuffle(available_types)
        results_pa = self.game.end_placement_phase()
        self.is_fast_mode_training = False
        return results_pa[3], results_pa[4], results_pa[5]

    def execute_boss_turn_for_training(self):
        return self.game.process_boss_attack()

    def execute_next_round_for_training(self):
        self.reset_all_round_animations()
        return self.game.proceed_to_next_round()

    def execute_boss_turn(self):
        self.log_message("Boss is thinking...")
        results = self.game.process_boss_attack()
        status_code, message, animation_triggers = results[0], results[1], results[2]
        for trigger in animation_triggers:
            anim_type=trigger["type"]
            if anim_type=="normal_attack":
                for r_target,c_target in trigger["targets"]: self.start_cell_persistent_flash(r_target,c_target,color1="gold",color2="#FFD700")
            elif anim_type=="ultimate_hit":
                 for r_target,c_target in trigger["targets"]: self.start_cell_persistent_flash(r_target,c_target,color1="orangered",color2="crimson")
            elif anim_type=="horizontal_shot": self.animate_sequential_shot(trigger["targets"],highlight_color="sandybrown",persist_color="peru")
            elif anim_type=="vertical_shot": self.animate_sequential_shot(trigger["targets"],highlight_color="lightskyblue",persist_color="steelblue")
            elif anim_type=="boss_heal": self.animate_boss_display_glow()
        self.set_controls_for_phase(self.game.game_phase); self.update_all_ui_displays()
        if status_code == "game_over_player_wiped": self.handle_game_over(message); return
        QTimer.singleShot(0 if self.is_fast_mode_training else 1000, self.execute_end_of_round)

    def execute_end_of_round(self):
        self.log_message("Round ending...")
        self.reset_all_round_animations()
        status_code, message, _ = self.game.proceed_to_next_round()
        self.set_controls_for_phase(self.game.game_phase)
        self.selected_unit_type_for_placement = None
        self.update_all_ui_displays(); self.log_message(message)
        if status_code == "game_over": self.handle_game_over(message)

    def handle_game_over(self, message):
        if not self.is_fast_mode_training : self.log_message(f"GAME OVER: {message}")
        self.set_controls_for_phase("GAME_OVER")
        self.reset_all_round_animations(); self.update_all_ui_displays()
        if not self.is_fast_mode_training :
            reply = QMessageBox.question(self,'Game Over',message+"\nPlay again?",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
            if reply == QMessageBox.Yes: self.start_new_game_ui()
            else: self.close()

def run_training_loop(window, agent, num_episodes):
    window.is_fast_mode_training = True
    all_episode_rewards = []
    recent_outcomes = [] 
    
    # Mở file CSV để ghi, nếu file không tồn tại, tạo mới và ghi header
    file_exists = os.path.isfile(TRAINING_STATS_FILE)
    with open(TRAINING_STATS_FILE, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists: # Ghi header nếu file mới được tạo
            csv_writer.writerow(['Episode', 'AvgReward', 'WinRate', 'Epsilon'])

        for e in range(num_episodes):
            window.current_episode_count = e + 1
            current_state_dict = window.game.start_new_game()
            window.start_new_game_ui(initial_state_dict_for_agent=current_state_dict)
            episode_reward = 0; done = False; boss_won_episode = False

            for game_round_num in range(window.game.max_rounds + 2):
                if done: break
                if (e * window.game.max_rounds + game_round_num) % 200 == 0:
                     window.update_all_ui_displays()
                else:
                     window.episode_label.setText(f"Episode: {window.current_episode_count}/{num_episodes} | R: {window.game.current_round}")
                     QApplication.processEvents()

                state_dict_after_player, reward_player_phase, done_player_phase = window.execute_player_turn_for_training()
                episode_reward += reward_player_phase
                if done_player_phase:
                    boss_won_episode = False
                    break
                
                current_state_dict_for_boss = state_dict_after_player
                boss_turn_results = window.execute_boss_turn_for_training()
                status_ui_boss_turn, _msg_ui, _anim, next_state_dict_after_boss, reward_for_boss_this_action, done_after_boss, state_dict_boss_acted_on, action_idx_boss_took = boss_turn_results
                episode_reward += reward_for_boss_this_action

                if action_idx_boss_took is not None and state_dict_boss_acted_on is not None:
                    current_discrete_state = agent._discretize_state(state_dict_boss_acted_on)
                    next_discrete_state = agent._discretize_state(next_state_dict_after_boss)
                    agent.learn(current_discrete_state, action_idx_boss_took, reward_for_boss_this_action, next_discrete_state, done_after_boss)
                
                current_state_dict = next_state_dict_after_boss
                done = done_after_boss

                if done:
                    if status_ui_boss_turn == "game_over_player_wiped": boss_won_episode = True
                    elif window.game.boss.current_hp <= 0 : boss_won_episode = False
                    elif window.game.boss.current_hp > 0 : boss_won_episode = True
                    break
                
                if window.game.current_round < window.game.max_rounds:
                    status_nr, msg_nr, next_state_dict_new_round = window.execute_next_round_for_training()
                    current_state_dict = next_state_dict_new_round
                    if status_nr == "game_over":
                        if window.game.boss.current_hp > 0: boss_won_episode = True
                        else: boss_won_episode = False
                        done = True; break
                else:
                    if not done :
                        if window.game.boss.current_hp > 0: boss_won_episode = True
                        else: boss_won_episode = False
                    done = True; break
            
            all_episode_rewards.append(episode_reward)
            recent_outcomes.append(1 if boss_won_episode else 0)
            if len(recent_outcomes) > LOG_STATS_EVERY_N_EPISODES:
                recent_outcomes.pop(0)

            if (e + 1) % LOG_STATS_EVERY_N_EPISODES == 0:
                avg_reward = sum(all_episode_rewards[-LOG_STATS_EVERY_N_EPISODES:]) / len(all_episode_rewards[-LOG_STATS_EVERY_N_EPISODES:])
                win_rate = sum(recent_outcomes) / len(recent_outcomes) * 100 if recent_outcomes else 0
                
                log_str = f"Ep {e+1}/{num_episodes}. Avg Reward (last {LOG_STATS_EVERY_N_EPISODES}): {avg_reward:.2f}. Win Rate: {win_rate:.1f}%. Epsilon: {agent.epsilon:.4f}"
                print(log_str) # In ra console
                window.log_message(log_str) # Vẫn log vào game UI nếu muốn
                
                # Ghi vào file CSV
                csv_writer.writerow([e + 1, f"{avg_reward:.2f}", f"{win_rate:.1f}", f"{agent.epsilon:.4f}"])
                csvfile.flush() # Đảm bảo dữ liệu được ghi ngay lập tức (tùy chọn)
            
            if (e + 1) % SAVE_AGENT_EVERY_N_EPISODES == 0:
                agent.save(f"q_table_boss_episode_{e+1}.pkl")
                print(f"Agent saved at episode {e+1}")
                # window.log_message(f"Agent saved at episode {e+1}")
    
    window.is_fast_mode_training = False
    agent.save(AGENT_MODEL_FILE)
    print(f"Training finished. Agent saved to {AGENT_MODEL_FILE}")
    # window.log_message(f"Training finished. Agent saved to {AGENT_MODEL_FILE}")


def main():
    app = QApplication(sys.argv)
    q_agent = QLearningTableAgent()
    if not TRAIN_MODE and os.path.exists(AGENT_MODEL_FILE):
        q_agent.load(AGENT_MODEL_FILE)
        q_agent.epsilon = 0.0
        print(f"Q-Table Agent model loaded from {AGENT_MODEL_FILE}")
    window = TacticsGridWindow(agent_to_use=q_agent)
    window.show()
    if TRAIN_MODE:
        print("Starting Q-Table training loop...")
        run_training_loop(window, q_agent, NUM_EPISODES_TO_TRAIN)
        window.is_fast_mode_training = False
        q_agent.epsilon = 0.0
        window.current_episode_count = 0
        if hasattr(window, 'episode_label'): window.episode_label.hide()
        window.start_new_game_ui()
        QMessageBox.information(window, "Training Complete", "Q-Table Agent training finished. Play against trained Boss.")
    else:
        window.start_new_game_ui()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()