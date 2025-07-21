# units.py

class Unit:
    def __init__(self, name, max_hp, current_hp, attack_power, abbr, position=None):
        self.name = name
        self.max_hp = max_hp
        self.current_hp = current_hp
        self.attack_power = attack_power
        self.abbr = abbr
        self.position = position

    def take_damage(self, amount):
        self.current_hp -= amount
        if self.current_hp < 0:
            self.current_hp = 0
        return self.current_hp <= 0

    def heal(self, amount):
        self.current_hp += amount
        if self.current_hp > self.max_hp:
            self.current_hp = self.max_hp

    def get_display_text(self):
        return f"{self.abbr}\n{self.current_hp}/{self.max_hp}"

    def __str__(self):
        return f"{self.name} ({self.current_hp}/{self.max_hp} HP, {self.attack_power} ATK)"

class Tank(Unit):
    def __init__(self, position=None):
        super().__init__("Tank", 3, 3, 0, "T", position)

class Knight(Unit):
    def __init__(self, position=None):
        super().__init__("Knight", 2, 2, 1, "K", position)

class AD(Unit):
    def __init__(self, position=None):
        super().__init__("AD", 1, 1, 2, "A", position)

PLAYER_UNIT_SPECS = {
    "Tank": {"class": Tank, "max_accumulation": 2, "abbr": "T"},
    "Knight": {"class": Knight, "max_accumulation": 2, "abbr": "K"},
    "AD": {"class": AD, "max_accumulation": 3, "abbr": "A"}
}