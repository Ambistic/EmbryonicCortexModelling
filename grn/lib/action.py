from enum import Enum


class Action(Enum):
    NoOp = 0
    Divide = 1
    Die = 2
    DiffNeuron = 3
    Migrate = 4