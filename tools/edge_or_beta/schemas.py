"""
ResultBundle Schema validation functions.
Ensures that the output of evaluate_rule matches the required JSON contract.
"""
from typing import Dict

REQUIRED_KEYS = {
    "schema_version",
    "run_id",
    "created_at",
    "rule",
    "settings",
    "data_quality",
    "strategy",
    "benchmarks",
    "decomposition",
    "verdict"
}

def validate_result_bundle(bundle: Dict) -> bool:
    """
    Validates that a result bundle has all required root keys.
    """
    missing = REQUIRED_KEYS - set(bundle.keys())
    if missing:
        raise ValueError(f"ResultBundle is missing required keys: {missing}")
    return True
