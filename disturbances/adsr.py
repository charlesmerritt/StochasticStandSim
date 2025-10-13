from __future__ import annotations


class ADSR:
    """Simple ADSR envelope to model post-disturbance effects."""
    def __init__(self, attack: float, decay: float, sustain: float, release: float):
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.level = 0.0
        self.active = False

def trigger(self):
    self.level = self.attack
    self.active = True

def step(self) -> float:
    if not self.active:
        return 0.0
    if self.level > self.sustain:
        self.level = max(self.sustain, self.level - self.decay)
    else:
        self.level = max(0.0, self.level - self.release)
    if self.level == 0.0:
        self.active = False
        return self.level