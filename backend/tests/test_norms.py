import pytest
from pydantic import TypeAdapter, ValidationError

from genfishery.models.norms import (
    AnyNormPrimitive,
    AssignRolePrimitive,
    CapPrimitive,
    NormSpec,
    PenalisePrimitive,
)


def test_default_penalise_has_sensible_defaults():
    p = PenalisePrimitive(id="p1", scope="collective")
    assert p.trigger == "exceed_cap"
    assert p.penalty_type == "forfeit"
    assert p.destination == "pool"


def test_norm_spec_compiles_multiple_linked_primitives():
    spec = NormSpec(
        id="norm-1",
        raw_text="No one may catch more than 20 units, and a monitor is elected to enforce it.",
        adopted_round=3,
        primitives=[
            CapPrimitive(id="p1", scope="collective", basis="fixed_units", value=20.0),
            AssignRolePrimitive(id="p2", scope="collective", role_name="monitor", selection="elected"),
        ],
    )
    assert isinstance(spec.primitives[0], CapPrimitive)
    assert isinstance(spec.primitives[1], AssignRolePrimitive)


def test_discriminated_union_round_trips_via_type_field():
    adapter = TypeAdapter(AnyNormPrimitive)
    raw = {
        "id": "p1",
        "scope": "individual",
        "type": "cap",
        "basis": "fixed_units",
        "value": 20.0,
    }
    parsed = adapter.validate_python(raw)
    assert parsed.type == "cap"
    assert parsed.value == 20.0


def test_unknown_type_discriminator_rejected():
    adapter = TypeAdapter(AnyNormPrimitive)
    with pytest.raises(ValidationError):
        adapter.validate_python({"id": "p1", "scope": "individual", "type": "not_a_primitive"})


def test_penalise_gated_to_monitor_role_via_trigger():
    amended = PenalisePrimitive(id="pen-amended", scope="collective", trigger="fail_monitor_duty")
    assert amended.trigger == "fail_monitor_duty"
    # Other defaults are untouched by changing the trigger.
    assert amended.penalty_type == "forfeit"
    assert amended.destination == "pool"
