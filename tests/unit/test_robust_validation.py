from modules.evaluation.robust_validation import MonteCarlo, WalkForward


def test_imports():
    assert MonteCarlo is not None
    assert WalkForward is not None


def test_montecarlo_class_exists():
    assert MonteCarlo is not None


def test_walkforward_class_exists():
    assert WalkForward is not None