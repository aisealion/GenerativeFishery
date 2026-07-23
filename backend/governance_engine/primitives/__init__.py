from .adjust import AdjustPrimitive
from .assign_role import AssignRolePrimitive
from .cap import CapPrimitive
from .declare import DeclarePrimitive
from .monitor import MonitorPrimitive
from .penalise import PenalisePrimitive
from .redistribute import RedistributePrimitive

PRIMITIVE_REGISTRY: dict = {
    "cap": CapPrimitive,
    "declare": DeclarePrimitive,
    "monitor": MonitorPrimitive,
    "penalise": PenalisePrimitive,
    "adjust": AdjustPrimitive,
    "redistribute": RedistributePrimitive,
    "assign_role": AssignRolePrimitive,
}

__all__ = [
    "CapPrimitive",
    "DeclarePrimitive",
    "MonitorPrimitive",
    "PenalisePrimitive",
    "AdjustPrimitive",
    "RedistributePrimitive",
    "AssignRolePrimitive",
    "PRIMITIVE_REGISTRY",
]
