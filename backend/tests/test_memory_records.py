import pytest

from genfishery.memory.records import decode, encode


def test_encode_decode_round_trip():
    text = encode(seq=3, round=5, importance=7.5, kind="observation", content="I harvested 4 fish.")
    record = decode(text)
    assert record.seq == 3
    assert record.round == 5
    assert record.importance == pytest.approx(7.5)
    assert record.kind == "observation"
    assert record.content == "I harvested 4 fish."


def test_encode_strips_newlines_from_content():
    text = encode(seq=0, round=0, importance=1.0, kind="plan", content="line one\nline two")
    record = decode(text)
    assert "\n" not in record.content
    assert record.content == "line one line two"


def test_decode_rejects_malformed_text():
    with pytest.raises(ValueError):
        decode("just some plain text with no metadata prefix")
