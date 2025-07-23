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
from agent import DQNAgent, get_game_state_for_q_table # Changed QLearningTableAgent to DQNAgent

# --- RL Agent Configuration ---
TRAIN_MODE = False # Set to True to enable training
NUM_EPISODES_TO_TRAIN = 200000
SAVE_AGENT_EVERY_N_EPISODES = 5000
AGENT_MODEL_FILE = "Model/dqn_boss_agent.pth" # Changed filename for DQN model
LOG_STATS_EVERY_N_EPISODES = 500
TRAINING_STATS_FILE = "Model/training_stats.csv" # CSV file to save training statistics

class TacticsGridWindow(QMainWindow):
    def __init__(self, agent_to_use=None):
        super().__init__()
        self.setWindowTitle("Tactics Grid – Boss Assault (DQN RL)") # Updated window title
        self.setGeometry(100, 100, 900, 850)
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.forest_bg_path = os.path.join(base_path, "assets", "forest_bg.png").replace("\\", "/")
        self.boss_gif_path = os.path.join(base_path, "assets", "boss_idle.gif").replace("\\", "/")
        
        # Pass the agent instance to GameLogic
        self.game = GameLogic(agent_instance=agent_to_use)
        
        # Ensure the agent has a reference to boss skills if needed (e.g., for exploration strategy)
        if agent_to_use and hasattr(self.game.boss, 'skills') and not agent_to_use.boss_skills_ref :
            agent_to_use.boss_skills_ref = self.game.boss.skills

        self.grid_buttons = [[None for _ in range(self.game.grid_size)] for _ in range(self.game.grid_size)]
        self.selected_unit_type_for_placement = None
        self.player_stock_buttons = {}
        self.cell_end_of_round_effects = {}
        self.boss_display_effect = {"type": None, "timer": None, "original_style": ""}
        self.short_term_animation_timers = []
        self.is_fast_mode_training = False # Flag for fast training mode (no UI updates)
        self.current_episode_count = 0

        self.init_ui()

    def init_ui(self):
        palette = QPalette()
        pixmap = QPixmap(self.forest_bg_path)
        if not pixmap.isNull():
            palette.setBrush(QPalette.Window, QBrush(pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)))
            self.setPalette(palette)
            self.setAutoFillBackground(True)
        else:
            self.setStyleSheet("QMainWindow { background-color: #2c3e50; }")

        self.central_widget = QWidget()
        self.central_widget.setAutoFillBackground(False) # Let background come from QMainWindow palette
        self.central_widget.setStyleSheet("background: transparent;")
        self.setCentralWidget(self.central_widget)

        overall_horizontal_layout = QHBoxLayout(self.central_widget)
        overall_horizontal_layout.addStretch(1) # Left stretch

        self.main_content_container = QWidget()
        self.main_content_container.setStyleSheet("background-color: rgba(20,20,30,0.75); border-radius:10px; padding:15px;")
        self.main_content_container.setFixedWidth(700) # Fixed width for content area

        self.main_content_v_layout = QVBoxLayout(self.main_content_container)
        overall_horizontal_layout.addWidget(self.main_content_container)
        overall_horizontal_layout.addStretch(1) # Right stretch

        # Episode Label (for training mode)
        self.episode_label = QLabel("Episode: 0")
        self.episode_label.setFont(QFont("Arial",10,QFont.Bold))
        self.episode_label.setStyleSheet("color: #FFD700; background:transparent;")
        self.episode_label.setAlignment(Qt.AlignRight)
        if not TRAIN_MODE: self.episode_label.hide() # Hide if not in training mode
        self.main_content_v_layout.addWidget(self.episode_label)

        # Round Label
        self.round_label = QLabel()
        self.round_label.setFont(QFont("Arial",16,QFont.Bold))
        self.round_label.setAlignment(Qt.AlignCenter)
        self.round_label.setStyleSheet("color:white; background:transparent;")
        self.main_content_v_layout.addWidget(self.round_label)

        # Player Stock Area
        stock_area_widget = QWidget()
        stock_area_layout = QVBoxLayout(stock_area_widget)
        stock_area_layout.setContentsMargins(0,0,0,0)

        self.player_stock_info_label = QLabel("Player Unit Stock (Click to select, then click grid):")
        self.player_stock_info_label.setFont(QFont("Arial",11,QFont.Bold))
        self.player_stock_info_label.setStyleSheet("color:#E0E0E0; background:transparent; margin-bottom:5px;")
        self.player_stock_info_label.setAlignment(Qt.AlignCenter)
        stock_area_layout.addWidget(self.player_stock_info_label)

        self.player_stock_buttons_layout = QHBoxLayout()
        for unit_name, spec in PLAYER_UNIT_SPECS.items():
            button = QPushButton(f"{unit_name}\n({spec['abbr']}) 0/0")
            button.setFont(QFont("Arial",10))
            button.setMinimumHeight(60)
            button.setCheckable(True) # Make buttons checkable for selection
            button.clicked.connect(lambda chk, name=unit_name: self.on_stock_unit_selected(name))
            self.player_stock_buttons[unit_name] = button
            self.player_stock_buttons_layout.addWidget(button)
        stock_area_layout.addLayout(self.player_stock_buttons_layout)
        self.main_content_v_layout.addWidget(stock_area_widget)

        # Boss Display Area
        boss_display_container = QWidget()
        boss_display_container.setStyleSheet("background:transparent;")
        boss_display_layout = QVBoxLayout(boss_display_container)
        boss_display_layout.setContentsMargins(0,5,0,5)
        boss_display_layout.setSpacing(5)

        self.boss_gif_label = QLabel()
        self.boss_gif_label.setAlignment(Qt.AlignCenter)
        boss_gif_size = 160
        self.boss_gif_label.setFixedSize(boss_gif_size,boss_gif_size)
        self.boss_display_effect["original_style"] = "background-color:rgba(50,50,70,0.3); border:1px solid #444; border-radius:8px;"
        self.boss_gif_label.setStyleSheet(self.boss_display_effect["original_style"])

        if os.path.exists(self.boss_gif_path):
            self.boss_movie = QMovie(self.boss_gif_path)
            if self.boss_movie.isValid():
                self.boss_movie.setScaledSize(QSize(boss_gif_size-10,boss_gif_size-10))
                self.boss_gif_label.setMovie(self.boss_movie)
                self.boss_movie.start()
            else:
                self.boss_gif_label.setText("Boss GIF\nInvalid")
        else:
            self.boss_gif_label.setText("Boss GIF\nNot Found")
        boss_display_layout.addWidget(self.boss_gif_label,alignment=Qt.AlignCenter)

        self.boss_hp_label = QLabel()
        self.boss_hp_label.setFont(QFont("Arial",12,QFont.Bold))
        self.boss_hp_label.setAlignment(Qt.AlignCenter)
        self.boss_hp_label.setStyleSheet("color:white; background:transparent; padding:3px;")
        boss_display_layout.addWidget(self.boss_hp_label,alignment=Qt.AlignCenter)
        self.main_content_v_layout.addWidget(boss_display_container,alignment=Qt.AlignCenter)

        # Action Log Display
        self.action_log_display = QTextEdit()
        self.action_log_display.setFont(QFont("Arial",10))
        self.action_log_display.setReadOnly(True)
        self.action_log_display.setFixedHeight(100)
        self.action_log_display.setStyleSheet("background-color:rgba(10,10,20,0.7); color:#E0E0E0; border-radius:5px; padding:5px; border:1px solid #333;")
        self.main_content_v_layout.addWidget(self.action_log_display)

        # Grid Layout
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(5)
        grid_frame = QFrame()
        grid_frame.setStyleSheet("background:transparent;")
        grid_frame.setLayout(self.grid_layout)

        for r in range(self.game.grid_size):
            for c in range(self.game.grid_size):
                button = QPushButton("")
                button.setFixedSize(80,80)
                button.setFont(QFont("Arial",10))
                button.clicked.connect(lambda checked, r_val=r, c_val=c: self.on_grid_cell_clicked(r_val,c_val))
                self.grid_buttons[r][c] = button
                self.grid_layout.addWidget(button,r,c)
        self.main_content_v_layout.addWidget(grid_frame,alignment=Qt.AlignCenter)

        # Placement Info Label
        self.placement_info_label = QLabel(f"Can place 0 more units this round.")
        self.placement_info_label.setFont(QFont("Arial",11,QFont.Bold))
        self.placement_info_label.setStyleSheet("color:#DDD; background:transparent; margin-top:5px;")
        self.placement_info_label.setAlignment(Qt.AlignCenter)
        self.main_content_v_layout.addWidget(self.placement_info_label)

        # Control Buttons
        self.control_layout = QHBoxLayout()
        self.end_placement_button = QPushButton("End Placement & Attack")
        self.end_placement_button.setFont(QFont("Arial",12,QFont.Bold))
        self.end_placement_button.setMinimumHeight(40)
        self.end_placement_button.setStyleSheet("QPushButton { background-color:#007BFF; color:white; border-radius:5px; padding:5px; } QPushButton:hover { background-color:#0056b3; } QPushButton:disabled { background-color:#555; color:#aaa; }")
        self.end_placement_button.clicked.connect(self.on_end_placement_clicked)
        self.control_layout.addWidget(self.end_placement_button)
        self.main_content_v_layout.addLayout(self.control_layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        palette = QPalette()
        pixmap = QPixmap(self.forest_bg_path)
        if not pixmap.isNull():
            palette.setBrush(QPalette.Window, QBrush(pixmap.scaled(self.size(),Qt.KeepAspectRatioByExpanding,Qt.SmoothTransformation)))
            self.setPalette(palette)

    def start_new_game_ui(self, initial_state_dict_for_agent=None):
        self.reset_all_round_animations()
        if initial_state_dict_for_agent is None : # For manual play or first episode of training
            self.game.start_new_game()
        # For training, game.start_new_game() is called directly in run_training_loop
        # and its initial state is passed here via initial_state_dict_for_agent.
        
        self.selected_unit_type_for_placement = None
        self.set_controls_for_phase(self.game.game_phase)
        self.update_all_ui_displays()
        if self.current_episode_count == 0 or not TRAIN_MODE:
            self.log_message("New Game Started. Select units from stock to place.")
        return get_game_state_for_q_table(self.game) # Return the dictionary, agent will discretize it

    def update_stock_buttons_display(self):
        can_place_more=self.game.can_place_more_units_this_round()
        for unit_name,button in self.player_stock_buttons.items():
            spec=PLAYER_UNIT_SPECS[unit_name]
            current_stock=self.game.player_current_accumulation.get(unit_name,0)
            max_stock=self.game.player_max_accumulation.get(unit_name,0)
            button.setText(f"{unit_name} ({spec['abbr']})\n{current_stock}/{max_stock}")
            
            # Button enabled only if there's stock, can place more units, and in placement phase
            button_enabled=current_stock>0 and can_place_more and self.game.game_phase=="PLACEMENT"
            button.setEnabled(button_enabled)

            # Style for selected vs unselected
            if self.selected_unit_type_for_placement==unit_name:
                button.setStyleSheet("QPushButton { background-color:lightgreen; color:black; border:2px solid green; padding:5px;} QPushButton:disabled { background-color:#444; color:#888; }")
            else:
                button.setStyleSheet("QPushButton { background-color:#555; border:1px solid #777; padding:5px; color:white;} QPushButton:hover { background-color:#666; } QPushButton:disabled { background-color:#444; color:#888; }")

    def on_stock_unit_selected(self,unit_name):
        if self.is_fast_mode_training and self.game.game_phase=="PLACEMENT":
            # In fast training, player actions are automated. No need for manual selection.
            if self.game.can_place_more_units_this_round() and self.game.player_current_accumulation.get(unit_name,0)>0:
                empty_cells=[(r,c_grid) for r in range(self.game.grid_size) for c_grid in range(self.game.grid_size) if self.game.grid_units[r][c_grid] is None]
                if empty_cells:
                    r_place,c_place=random.choice(empty_cells)
                    self.game.place_unit_from_stock(unit_name,r_place,c_place)
                    self.update_all_ui_displays() # Update UI in fast mode for batch updates
            return

        if not self.game.can_place_more_units_this_round() and self.selected_unit_type_for_placement!=unit_name:
            self.log_message("Placement limit reached.",is_error=True)
            if self.selected_unit_type_for_placement is not None: # If already selected, allow deselection
                pass
            else: # If trying to select a new one, but limit reached, clear selection
                self.selected_unit_type_for_placement=None
            self.update_stock_buttons_display()
            return

        if self.game.player_current_accumulation.get(unit_name,0)<=0 and self.selected_unit_type_for_placement!=unit_name:
            self.log_message(f"No {unit_name}s left.",is_error=True)
            self.update_stock_buttons_display()
            return

        # Toggle selection
        if self.selected_unit_type_for_placement==unit_name:
            self.selected_unit_type_for_placement=None
            self.log_message(f"Deselected {unit_name}.")
        else:
            if not self.game.can_place_more_units_this_round():
                self.log_message("Placement limit. Cannot select new.",is_error=True)
            elif self.game.player_current_accumulation.get(unit_name,0)<=0:
                self.log_message(f"No {unit_name}s left to select.",is_error=True)
            else:
                self.selected_unit_type_for_placement=unit_name
                self.log_message(f"Selected {unit_name}. Click grid.")
        self.update_stock_buttons_display()


    def reset_all_round_animations(self):
        # Stop and clear all short-term animation timers
        for t in self.short_term_animation_timers:
            t.stop()
            t.deleteLater()
        self.short_term_animation_timers.clear()

        # Stop and clear all cell-specific end-of-round effects
        for k,v in list(self.cell_end_of_round_effects.items()):
            if v.get("timer"):
                v["timer"].stop()
                v["timer"].deleteLater()
        self.cell_end_of_round_effects.clear()

        # Reset boss display effects
        if self.boss_display_effect.get("timer"):
            self.boss_display_effect["timer"].stop()
            self.boss_display_effect["timer"].deleteLater()
        if hasattr(self,'boss_gif_label') and self.boss_gif_label:
            self.boss_gif_label.setStyleSheet(self.boss_display_effect["original_style"])
        self.boss_display_effect["type"]=None
        self.boss_display_effect["timer"]=None


    def get_cell_font_weight_style(self,r,c):
        return "font-weight:bold; color:black;" if self.game.grid_units[r][c] else "font-weight:normal; color:black;"

    def animate_boss_display_glow(self,color="rgba(144,238,144,0.6)"):
        # Stop any existing boss animation timer
        if self.boss_display_effect.get("timer"):
            self.boss_display_effect["timer"].stop()
            self.boss_display_effect["timer"]=None
        
        style=f"background-color:{color}; border:2px solid lightgreen; border-radius:8px;"
        if hasattr(self,'boss_gif_label') and self.boss_gif_label:
            self.boss_gif_label.setStyleSheet(style)
        self.boss_display_effect["type"]="persistent_glow"

    def animate_boss_display_persistent_flash(self,color1="rgba(255,0,0,0.5)",color2="rgba(255,127,127,0.5)",interval=300):
        if hasattr(self,'boss_gif_label') and self.boss_gif_label:
            # If a glow effect is active, reset it first
            if self.boss_display_effect.get("type")=="persistent_glow":
                self.boss_gif_label.setStyleSheet(self.boss_display_effect["original_style"])
            
            # Stop any existing flash timer
            if self.boss_display_effect.get("timer"):
                self.boss_display_effect["timer"].stop()

            flash_state={"is_color1":True}
            def flash():
                if not self.boss_gif_label: return # Guard against widget being destroyed
                bg_color=color1 if flash_state["is_color1"] else color2
                style=f"background-color:{bg_color}; border:2px solid red; border-radius:8px;"
                self.boss_gif_label.setStyleSheet(style)
                flash_state["is_color1"]=not flash_state["is_color1"]

            timer=QTimer(self)
            timer.timeout.connect(flash)
            timer.start(interval)
            self.boss_display_effect["type"]="persistent_flash"
            self.boss_display_effect["timer"]=timer
            flash() # Call once immediately

    def start_cell_persistent_flash(self,r,c,color1="yellow",color2="lightcoral",interval=350):
        # Stop any existing effect on this cell
        if (r,c) in self.cell_end_of_round_effects and self.cell_end_of_round_effects[(r,c)].get("timer"):
            self.cell_end_of_round_effects[(r,c)]["timer"].stop()

        button=self.grid_buttons[r][c]
        font_style=self.get_cell_font_weight_style(r,c)
        flash_state={"is_color1":True}

        def flash():
            if not button: return # Guard against widget being destroyed
            bg_color=color1 if flash_state["is_color1"] else color2
            button.setStyleSheet(f"background-color:{bg_color}; {f_style} border-radius:5px;");flash_state["is_color1"]=not flash_state["is_color1"]

        timer=QTimer(self)
        timer.timeout.connect(flash)
        timer.start(interval)
        self.cell_end_of_round_effects[(r,c)]={"type":"persistent_flash","timer":timer,"data":{}}
        flash() # Call once immediately

    def animate_sequential_shot(self,targets,highlight_color="orange",persist_color="darkorange",sequential_delay=300,highlight_duration=250):
        if not targets: return

        for i,(r_coord,c_coord) in enumerate(targets):
            button=self.grid_buttons[r_coord][c_coord]
            font_style=self.get_cell_font_weight_style(r_coord,c_coord)

            def light_up_cell(b=button,r_cell=r_coord,c_cell=c_coord,f_style=font_style):
                if not b: return
                b.setStyleSheet(f"background-color:{highlight_color}; {f_style} border-radius:5px;")
                
                # Timer to revert/persist color after highlight_duration
                persist_timer=QTimer(self)
                persist_timer.setSingleShot(True)
                persist_timer.timeout.connect(lambda:self.set_cell_persistent_style(r_cell,c_cell,persist_color,f_style))
                persist_timer.start(highlight_duration)
                self.short_term_animation_timers.append(persist_timer)

            # Timer to trigger light_up_cell sequentially
            effect_timer=QTimer(self)
            effect_timer.setSingleShot(True)
            effect_timer.timeout.connect(light_up_cell)
            effect_timer.start(i*sequential_delay)
            self.short_term_animation_timers.append(effect_timer)

    def set_cell_persistent_style(self,r,c,color,font_style):
        button=self.grid_buttons[r][c]
        if not button: return
        button.setStyleSheet(f"background-color:{color}; {font_style} border-radius:5px;")
        # Store this as a persistent effect for the round
        self.cell_end_of_round_effects[(r,c)]={"type":"persistent_style","timer":None,"data":{"color":color}}


    def animate_tank_block(self,r,c,color1="white",color2="dodgerblue",interval=166,flashes=3):
        button=self.grid_buttons[r][c]
        original_unit_color="slategray" # Default Tank color
        font_style=self.get_cell_font_weight_style(r,c)
        if not button: return

        flash_state={"is_color1":True,"count":0,"max_flashes":flashes*2} # Each flash has two color states
        timer=QTimer(self)

        def flash():
            if not button:
                timer.stop()
                return
            
            flash_state["count"]+=1
            if flash_state["count"] > flash_state["max_flashes"]:
                button.setStyleSheet(f"background-color:{original_unit_color}; {font_style} border-radius:5px;");timer.stop()
                if timer in self.short_term_animation_timers: # Remove the timer from active list
                    self.short_term_animation_timers.remove(timer)
                timer.deleteLater() # Clean up timer object
                return
            
            bg_color=color1 if flash_state["is_color1"] else color2
            button.setStyleSheet(f"background-color:{bg_color}; {font_style} border-radius:5px;");flash_state["is_color1"]=not flash_state["is_color1"]

        timer.timeout.connect(flash)
        timer.start(interval)
        self.short_term_animation_timers.append(timer) # Add to list to manage cleanup
        flash() # Call once immediately

    def update_all_ui_displays(self):
        self.update_info_display()
        self.update_stock_buttons_display()
        self.update_grid_display()
        self.update_action_log_display()
        self.update_placement_info_label()
        
        if TRAIN_MODE and hasattr(self, 'episode_label'):
             # Only update episode label during fast training to reduce overhead
             self.episode_label.setText(f"Episode: {self.current_episode_count}/{NUM_EPISODES_TO_TRAIN} | R: {self.game.current_round}")
             if self.is_fast_mode_training:
                 # Process UI events periodically during fast training to prevent freezing
                 QApplication.processEvents()

    def update_info_display(self):
        self.round_label.setText(f"Round: {self.game.current_round}/{self.game.max_rounds}")
        boss_text = f"Boss HP: {self.game.boss.current_hp}/{self.game.boss.max_hp} | Rage: {self.game.boss.current_rage}"
        self.boss_hp_label.setText(boss_text)
        
        # Reset boss display style if no specific effect is active
        if self.boss_display_effect["type"] is None and hasattr(self,'boss_gif_label') and self.boss_gif_label:
             self.boss_gif_label.setStyleSheet(self.boss_display_effect["original_style"])

    def update_placement_info_label(self):
        if self.game.game_phase == "PLACEMENT":
            remaining = self.game.get_max_units_to_place_this_round() - self.game.units_placed_this_round_count
            self.placement_info_label.setText(f"Can place {remaining} more units this round.")
            self.placement_info_label.show()
        else:
            self.placement_info_label.hide()

    def update_grid_display(self):
        for r_idx in range(self.game.grid_size):
            for c_idx in range(self.game.grid_size):
                unit=self.game.grid_units[r_idx][c_idx]
                button=self.grid_buttons[r_idx][c_idx]
                font_style=self.get_cell_font_weight_style(r_idx,c_idx)

                if unit:
                    button.setText(unit.get_display_text())
                else:
                    button.setText("")
                
                # Apply end-of-round effect if exists, otherwise apply default unit color
                if (r_idx,c_idx) in self.cell_end_of_round_effects:
                    pass # Effect is managed by its own timer/logic
                else:
                    if unit:
                        color="gray" # Default color for units without specific assignment
                        if unit.name=="Tank": color="slategray"
                        elif unit.name=="Knight": color="lightblue"
                        elif unit.name=="AD": color="lightcoral"
                        button.setStyleSheet(f"background-color:{color}; {font_style} border-radius:5px;")
                    else: # Empty cell
                        button.setStyleSheet(f"background-color:rgba(80,80,80,0.7); {font_style} border-radius:5px;")

    def update_action_log_display(self):
        self.action_log_display.clear()
        # Show fewer logs in fast training mode for performance
        log_text = "\n".join(self.game.get_action_log(tail=5 if self.is_fast_mode_training else 15))
        self.action_log_display.setText(log_text)
        # Scroll to bottom
        self.action_log_display.verticalScrollBar().setValue(self.action_log_display.verticalScrollBar().maximum())

    def log_message(self, message, is_error=False, is_success=False):
        # In fast training mode, only log critical messages to the internal log.
        # Otherwise, log all messages.
        if not self.is_fast_mode_training:
            self.game.action_log.append(message)
        self.update_action_log_display()

    def on_grid_cell_clicked(self, r, c):
        if self.is_fast_mode_training and self.game.game_phase == "PLACEMENT":
            # Manual clicks are disabled during fast training mode
            return

        if self.game.game_phase != "PLACEMENT":
            self.log_message("Not in placement phase.", is_error=True)
            return
        
        if self.selected_unit_type_for_placement is None:
            self.log_message("Select unit type from stock first.", is_error=True)
            return

        unit_to_place = self.selected_unit_type_for_placement
        success, message = self.game.place_unit_from_stock(unit_to_place, r, c)
        
        self.log_message(message, is_error=not success, is_success=success)
        if success:
            self.selected_unit_type_for_placement = None # Deselect unit after placement
            self.update_all_ui_displays()
        else:
            self.update_stock_buttons_display()
            self.update_placement_info_label()

    def set_controls_for_phase(self, phase):
        is_placement = (phase == "PLACEMENT")
        self.end_placement_button.setEnabled(is_placement)
        
        for unit_name, button in self.player_stock_buttons.items():
            button_enabled = (is_placement and self.game.player_current_accumulation.get(unit_name,0)>0 and self.game.can_place_more_units_this_round())
            button.setEnabled(button_enabled)
            
        self.placement_info_label.setVisible(is_placement)

        for r_idx in range(self.game.grid_size):
            for c_idx in range(self.game.grid_size):
                if self.grid_buttons[r_idx][c_idx]:
                    self.grid_buttons[r_idx][c_idx].setEnabled(is_placement)

    def on_end_placement_clicked(self):
        if self.is_fast_mode_training and self.game.game_phase == "PLACEMENT": return # Skip if in fast training
        if self.game.game_phase != "PLACEMENT": return # Only allow ending placement in placement phase

        self.log_message("Player ends turn actions...")
        self.selected_unit_type_for_placement = None # Clear selection

        results = self.game.end_placement_phase()
        status_code, message, damage_to_boss = results[0], results[1], results[2]
        self.log_message(message)

        if damage_to_boss > 0 and status_code != "game_over_boss_defeated":
            self.animate_boss_display_persistent_flash() # Flash boss if damaged
        
        self.set_controls_for_phase(self.game.game_phase)
        self.update_all_ui_displays()

        if status_code == "game_over_boss_defeated":
            self.handle_game_over(message)
            return
        
        # Schedule boss turn after a delay
        QTimer.singleShot(0 if self.is_fast_mode_training else 1000, self.execute_boss_turn)

    # --- Training-specific execution functions ---
    def execute_player_turn_for_training(self):
        """Automates player turn for training without UI interaction."""
        self.is_fast_mode_training = True
        self.game.game_phase = "PLACEMENT" # Ensure game state is correct for internal logic
        
        num_to_place = self.game.get_max_units_to_place_this_round()
        placed_count = 0
        
        # Filter available unit types (those with stock) and shuffle them
        available_types = [utype for utype, count in self.game.player_current_accumulation.items() if count > 0]
        random.shuffle(available_types)

        for _ in range(num_to_place):
            if not self.game.can_place_more_units_this_round() or not available_types:
                break # Stop if placement limit reached or no units left to place
            
            unit_type = random.choice(available_types) # Choose a random available unit type

            if self.game.player_current_accumulation[unit_type] > 0:
                empty_cells = [(r,c) for r in range(self.game.grid_size) for c in range(self.game.grid_size) if self.game.grid_units[r][c] is None]
                if empty_cells:
                    r_place,c_place = random.choice(empty_cells)
                    success,_ = self.game.place_unit_from_stock(unit_type,r_place,c_place)
                    if success:
                        placed_count+=1
                        # If a unit type's stock becomes 0, remove it from available_types list
                        if self.game.player_current_accumulation[unit_type]==0:
                            if unit_type in available_types:
                                available_types.remove(unit_type)
            
            # If no available types are left but we still need to place units,
            # refresh available_types from current stock.
            if not available_types and placed_count < num_to_place:
                available_types = [utype for utype,count in self.game.player_current_accumulation.items() if count>0]
                if not available_types: break # If still no units, break
                else: random.shuffle(available_types) # Shuffle for next attempts

        results_pa = self.game.end_placement_phase() # Process player attack phase
        self.is_fast_mode_training = False # Exit fast mode (though immediately re-entered for boss turn)
        return results_pa[3], results_pa[4], results_pa[5] # next_state_dict, reward, done (for player phase)

    def execute_boss_turn_for_training(self):
        """Executes boss turn for training, no UI delays."""
        # This function directly calls game.process_boss_attack()
        # The return values are then used by the training loop to update the agent.
        return self.game.process_boss_attack()

    def execute_next_round_for_training(self):
        """Proceeds to next round for training, resets animations."""
        self.reset_all_round_animations() # Clear UI animations
        return self.game.proceed_to_next_round()

    # --- UI-driven execution functions (for non-training/playback) ---
    def execute_boss_turn(self):
        self.log_message("Boss is thinking...")
        
        results = self.game.process_boss_attack()
        status_code, message, animation_triggers = results[0], results[1], results[2]
        
        # Trigger UI animations based on boss skill
        for trigger in animation_triggers:
            anim_type=trigger["type"]
            if anim_type=="normal_attack":
                for r_target,c_target in trigger["targets"]:
                    self.start_cell_persistent_flash(r_target,c_target,color1="gold",color2="#FFD700")
            elif anim_type=="ultimate_hit":
                 for r_target,c_target in trigger["targets"]:
                     self.start_cell_persistent_flash(r_target,c_target,color1="orangered",color2="crimson")
            elif anim_type=="horizontal_shot":
                self.animate_sequential_shot(trigger["targets"],highlight_color="sandybrown",persist_color="peru")
            elif anim_type=="vertical_shot":
                self.animate_sequential_shot(trigger["targets"],highlight_color="lightskyblue",persist_color="steelblue")
            elif anim_type=="boss_heal":
                self.animate_boss_display_glow()

        self.set_controls_for_phase(self.game.game_phase)
        self.update_all_ui_displays()

        if status_code == "game_over_player_wiped":
            self.handle_game_over(message)
            return
        
        # Schedule end of round after a delay
        QTimer.singleShot(0 if self.is_fast_mode_training else 1000, self.execute_end_of_round)

    def execute_end_of_round(self):
        self.log_message("Round ending...")
        self.reset_all_round_animations() # Clear all visual effects from the round

        status_code, message, _ = self.game.proceed_to_next_round() # This handles game over check and new round setup
        
        self.set_controls_for_phase(self.game.game_phase)
        self.selected_unit_type_for_placement = None # Clear player's unit selection
        self.update_all_ui_displays()
        self.log_message(message)

        if status_code == "game_over":
            self.handle_game_over(message)

    def handle_game_over(self, message):
        if not self.is_fast_mode_training : # Only show QMessageBox if not in fast training
            self.log_message(f"GAME OVER: {message}")
        
        self.set_controls_for_phase("GAME_OVER") # Disable controls
        self.reset_all_round_animations()
        self.update_all_ui_displays()

        if not self.is_fast_mode_training :
            reply = QMessageBox.question(self,'Game Over',message+"\nPlay again?",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.start_new_game_ui() # Start a new game
            else:
                self.close() # Close the application

def run_training_loop(window, agent, num_episodes):
    """
    Main training loop for the DQN agent. Runs many episodes without UI delays.
    """
    window.is_fast_mode_training = True # Enable fast mode (minimal UI updates)
    all_episode_rewards = []
    recent_outcomes = [] # Stores 1 for boss win, 0 for player win

    # Open CSV file to log training statistics
    file_exists = os.path.isfile(TRAINING_STATS_FILE)
    with open(TRAINING_STATS_FILE, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists: # Write header if file is new
            csv_writer.writerow(['Episode', 'AvgReward', 'WinRate_Boss', 'Epsilon'])

        for e in range(num_episodes):
            window.current_episode_count = e + 1 # Update episode counter for UI label
            
            # Start a new game and get the initial state dictionary
            current_game_state_dict = window.game.start_new_game() 
            
            episode_reward = 0 # Accumulator for total reward in the current episode
            # No need for a local 'done' flag to track outcome inside the loop,
            # it's just for breaking the loop. The outcome will be determined at the end.

            # Simulate game rounds until max_rounds or game over
            for game_round_num in range(window.game.max_rounds + 2): # +2 for potential final round processing
                # Check definitive game over conditions
                is_game_over, _ = window.game.check_game_over_conditions()
                if is_game_over:
                    break # Game has ended, break out of round loop

                # Periodically update UI to show progress during fast training
                if (e * window.game.max_rounds + game_round_num) % 200 == 0:
                     window.update_all_ui_displays()
                else:
                     # Minimal UI update for performance (just episode label)
                     if hasattr(window, 'episode_label'):
                        window.episode_label.setText(f"Episode: {window.current_episode_count}/{num_episodes} | R: {window.game.current_round}")
                     QApplication.processEvents() # Process events to keep UI responsive

                # --- Player Turn (Automated) ---
                state_dict_after_player, reward_player_phase, done_player_phase = window.execute_player_turn_for_training()
                episode_reward += reward_player_phase # Add player phase reward

                if done_player_phase: # This means player defeated boss in their attack phase
                    # Game ended by player defeating boss, so break
                    break # Exit round loop

                # --- Boss Turn (Agent's Action) ---
                current_state_dict_for_boss_action = state_dict_after_player 

                boss_turn_results = window.execute_boss_turn_for_training()
                status_ui_boss_turn, _msg_ui, _anim, next_state_dict_after_boss, reward_for_boss_this_action, done_after_boss, state_dict_boss_acted_on, action_idx_boss_took = boss_turn_results
                episode_reward += reward_for_boss_this_action # Add boss phase reward

                # Agent learning step: current_state, action, reward, next_state, done
                if action_idx_boss_took is not None and state_dict_boss_acted_on is not None:
                    # Discretize state dictionaries into numpy arrays for the DQN
                    current_state_vector = agent._discretize_state(state_dict_boss_acted_on)
                    next_state_vector = agent._discretize_state(next_state_dict_after_boss)
                    agent.learn(current_state_vector, action_idx_boss_took, reward_for_boss_this_action, next_state_vector, done_after_boss)
                
                if done_after_boss: # Game over condition detected during boss turn
                    break # End episode

                # --- End of Round / Proceed to Next Round ---
                # This prepares for the next round. It also checks for game over conditions.
                status_nr, msg_nr, next_state_dict_new_round = window.execute_next_round_for_training()
                
                if status_nr == "game_over":
                    # Game over at the start of a new round (e.g., boss survives all rounds)
                    break # End episode
                # If current_round is max_rounds after increment, the game is implicitly over,
                # even if not explicitly caught by check_game_over_conditions yet.
                # The outer loop condition `range(window.game.max_rounds + 2)` will handle breaking.

            # --- DETERMINE EPISODE OUTCOME *AFTER* THE ROUND LOOP HAS BROKEN ---
            boss_won_episode = False # Default assumption: Player wins (Boss loses)

            # Use the definitive game state to determine the winner
            final_is_game_over, final_message = window.game.check_game_over_conditions()
            
            if final_is_game_over:
                if window.game.boss.current_hp <= 0:
                    boss_won_episode = False # Player won: Boss was defeated
                else: # Boss HP > 0, so boss survived all rounds or wiped player
                    boss_won_episode = True # Boss won
            else:
                # This 'else' block should ideally not be reached. If the loop broke,
                # `final_is_game_over` should be True. If it's not, it means the loop
                # broke for an unexpected reason or a game-ending condition wasn't set.
                # For robustness, we will assume boss lost if the game state is ambiguous.
                print(f"WARNING: Episode {e+1} ended without definitive game over conditions being met. Defaulting to boss loss. Current Round: {window.game.current_round}, Boss HP: {window.game.boss.current_hp}")
                boss_won_episode = False # Default to boss loss in unexpected scenario

            all_episode_rewards.append(episode_reward)
            recent_outcomes.append(1 if boss_won_episode else 0) # 1 if boss won, 0 if player won

            # Keep only the last LOG_STATS_EVERY_N_EPISODES outcomes
            if len(recent_outcomes) > LOG_STATS_EVERY_N_EPISODES:
                recent_outcomes.pop(0)

            # Log and save statistics periodically
            if (e + 1) % LOG_STATS_EVERY_N_EPISODES == 0:
                avg_reward = sum(all_episode_rewards[-LOG_STATS_EVERY_N_EPISODES:]) / len(all_episode_rewards[-LOG_STATS_EVERY_N_EPISODES:])
                win_rate_boss = sum(recent_outcomes) / len(recent_outcomes) * 100 if recent_outcomes else 0
                
                log_str = f"Ep {e+1}/{num_episodes}. Avg Reward (last {LOG_STATS_EVERY_N_EPISODES}): {avg_reward:.2f}. Win Rate (Boss): {win_rate_boss:.1f}%. Epsilon: {agent.epsilon:.4f}"
                print(log_str) # Print to console
                window.log_message(log_str) # Also log to UI action log

                # Write statistics to CSV
                csv_writer.writerow([e + 1, f"{avg_reward:.2f}", f"{win_rate_boss:.1f}", f"{agent.epsilon:.4f}"])
                csvfile.flush() # Ensure data is written immediately (optional)
            
            # Save agent model periodically
            if (e + 1) % SAVE_AGENT_EVERY_N_EPISODES == 0:
                agent.save(f"dqn_boss_episode_{e+1}.pth") # Save with episode number for checkpoints
                print(f"Agent saved at episode {e+1}")
                # window.log_message(f"Agent saved at episode {e+1}") # Uncomment if want this in UI log

    # Training finished. Save final model.
    window.is_fast_mode_training = False
    agent.save(AGENT_MODEL_FILE)
    print(f"Training finished. Agent saved to {AGENT_MODEL_FILE}")
    # window.log_message(f"Training finished. Agent saved to {AGENT_MODEL_FILE}") # Uncomment if want this in UI log


def main():
    app = QApplication(sys.argv)
    
    # Instantiate the DQNAgent
    dqn_agent = DQNAgent()

    # Load agent model if not in training mode and file exists
    if not TRAIN_MODE and os.path.exists(AGENT_MODEL_FILE):
        dqn_agent.load(AGENT_MODEL_FILE)
        dqn_agent.epsilon = 0.0 # No exploration when playing against a trained agent
        print(f"DQN Agent model loaded from {AGENT_MODEL_FILE}")

    # Create and show the game window, passing the agent instance
    window = TacticsGridWindow(agent_to_use=dqn_agent)
    window.show()

    if TRAIN_MODE:
        print("Starting DQN training loop...")
        run_training_loop(window, dqn_agent, NUM_EPISODES_TO_TRAIN)
        
        # After training, reset to evaluation mode and start a new game
        window.is_fast_mode_training = False
        dqn_agent.epsilon = 0.0 # Ensure epsilon is 0 for evaluation after training
        window.current_episode_count = 0
        if hasattr(window, 'episode_label'): window.episode_label.hide() # Hide episode label post-training
        window.start_new_game_ui() # Start a new game with the trained agent
        QMessageBox.information(window, "Training Complete", "DQN Agent training finished. Play against trained Boss.")
    else:
        window.start_new_game_ui() # Start a new game with the loaded (or untrained) agent

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()