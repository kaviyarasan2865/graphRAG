"""Unit tests for scoring metrics."""

from app.eval.metrics import answer_matches, completeness


def test_exact_name():
    assert answer_matches("Chen Ding", ["Chen Ding"])


def test_added_country_code():
    assert answer_matches("Chen Ding (CHN)", ["Chen Ding"])


def test_accent_insensitive():
    assert answer_matches("Naim Suleymanoglu", ["Naim Süleymanoğlu"])


def test_numeric_match():
    assert answer_matches("5", ["5"])
    assert answer_matches("The answer is 8.", ["8"])
    assert not answer_matches("2", ["5"])


def test_numeric_not_confused_by_other_numbers():
    # gold is numeric 8; predicting a different number fails
    assert not answer_matches("0", ["8"])


def test_wrong_name():
    assert not answer_matches("Someone Else", ["Chen Ding"])


def test_dash_normalization():
    gold = ["Athletics at the 2008 Summer Olympics – Men's marathon"]
    assert answer_matches("Athletics at the 2008 Summer Olympics - Men's marathon", gold)
    # partial (the distinctive part) also matches via containment
    assert answer_matches("Men's marathon", gold)


def test_completeness():
    assert completeness(["A", "B", "C"], ["A", "B"]) == 1.0
    assert completeness(["A"], ["A", "B"]) == 0.5
    assert completeness([], ["A", "B"]) == 0.0
    assert completeness([], []) == 1.0
