from modules.evaluation.robust_validation import MonteCarlo, WalkForward


def test_imports():
    assert MonteCarlo is not None
    assert WalkForward is not None