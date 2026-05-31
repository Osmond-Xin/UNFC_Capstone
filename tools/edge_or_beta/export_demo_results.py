import os
import json
import argparse
from tools.edge_or_beta.engine import evaluate_rule

def main():
    parser = argparse.ArgumentParser(description="Export precomputed demo results for Edge or Beta? tool")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/Users/osmond/Documents/Job/niagaradataanalyst/public/edge-or-beta/demo_results",
        help="Path to the frontend public demo_results folder"
    )
    parser.add_argument("--n-null", type=int, default=1000, help="Number of Monte Carlo iterations")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Exporting results to: {args.output_dir} with n_null={args.n_null}...")

    # Define the 3 presets to compute
    presets = [
        {
            "id": "capstone_expiry_rule",
            "params": {"rsi_threshold": 30, "min_consecutive": 3},
            "filename": "capstone_expiry_rule.json"
        },
        {
            "id": "rsi_oversold",
            "params": {"rsi_threshold": 30},
            "filename": "rsi_oversold.json"
        },
        {
            "id": "ma_crossover",
            "params": {"short_period": 9, "long_period": 50},
            "filename": "ma_crossover.json"
        }
    ]

    for preset in presets:
        print(f"\nEvaluating preset: {preset['id']}...")
        try:
            bundle = evaluate_rule(
                rule_id=preset["id"],
                params=preset["params"],
                start="2018-01-02", # Match SPY cache coverage (SPY.csv starts 2018-01-02)
                end="2025-06-30", # Align with capstone's end date in portfolio_simulator
                hold_days=6,
                commission=0.002,
                top_k=3,
                n_null=args.n_null,
                seed=args.seed
            )
            
            output_path = os.path.join(args.output_dir, preset["filename"])
            with open(output_path, "w") as f:
                json.dump(bundle, f, indent=2)
            print(f"✓ Saved {preset['id']} to {output_path}")
        except Exception as e:
            print(f"✗ Failed to evaluate {preset['id']}: {e}")

if __name__ == "__main__":
    main()
