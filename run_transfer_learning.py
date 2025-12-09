
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from src.pde.black_scholes import create_black_scholes_pinn
from src.pde.heat_equation import create_heat_equation_pinn
from src.training.trainer import (
    train_pinn,
    evaluate_heat_pinn,
    print_evaluation_results,
    transfer_weights
)


def run_transfer_learning_experiment():
    print("=" * 70)
    print("PHYSICS-INFORMED NEURAL NETWORK: TRANSFER LEARNING EXPERIMENT")
    print("=" * 70)
    print("\nTesting hypothesis: Do PINNs learn transferable mathematical structure?")
    print("Approach: Train on Black-Scholes → Transfer to Heat Equation")
    print("=" * 70)

    # Configuration
    bs_config = {
        'spot_range': (0.0, 400.0),
        'time_to_expiry': 0.02,
        'strike': 200.0,
        'volatility': 0.2,
        'risk_free_rate': 0.05,
        'dividend_rate': 0.03
    }

    heat_config = {
        'spatial_range': (0.0, 1.0),
        'time_range': (0.0, 0.1),
        'value_range': (-1.0, 1.0),
        'diffusivity': 0.01
    }

    # Step 1: Train Black-Scholes PINN
    print("\n" + "=" * 70)
    print("STEP 1: Training Black-Scholes PINN")
    print("=" * 70)

    bs_pinn = create_black_scholes_pinn(**bs_config)
    bs_losses = train_pinn(
        bs_pinn,
        epochs=8000,
        n_collocation=1000,
        print_freq=1000
    )

    # Step 2: Validate Black-Scholes (optional - requires FD reference)
    print("\n" + "=" * 70)
    print("STEP 2: Black-Scholes Validation")
    print("=" * 70)
    print("Black-Scholes PINN trained successfully.")
    print("(For validation against finite difference, run experiments/run_fd_verification.py)")

    # Step 3: Transfer to Heat Equation
    print("\n" + "=" * 70)
    print("STEP 3: Transfer Learning → Heat Equation")
    print("=" * 70)

    heat_pinn_transfer = create_heat_equation_pinn(**heat_config)
    transfer_weights(bs_pinn, heat_pinn_transfer)

    transfer_losses = train_pinn(
        heat_pinn_transfer,
        epochs=3000,
        n_collocation=1000,
        print_freq=600
    )

    # Step 4: Train Fresh Heat PINN
    print("\n" + "=" * 70)
    print("STEP 4: Training Fresh Heat PINN (No Transfer)")
    print("=" * 70)

    heat_pinn_fresh = create_heat_equation_pinn(**heat_config)
    fresh_losses = train_pinn(
        heat_pinn_fresh,
        epochs=5000,
        n_collocation=1000,
        print_freq=1000
    )

    # Step 5: Evaluate Both
    print("\n" + "=" * 70)
    print("STEP 5: Evaluation Against Analytical Solution")
    print("=" * 70)

    fresh_errors = evaluate_heat_pinn(
        heat_pinn_fresh,
        test_times=[0.02, 0.05, 0.08],
        diffusivity=heat_config['diffusivity']
    )

    transfer_errors = evaluate_heat_pinn(
        heat_pinn_transfer,
        test_times=[0.02, 0.05, 0.08],
        diffusivity=heat_config['diffusivity']
    )

    print_evaluation_results(fresh_errors, "Fresh Heat PINN (No Transfer)")
    print_evaluation_results(transfer_errors, "Transfer Heat PINN (BS → Heat)")

    # Step 6: Results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    fresh_l2 = fresh_errors['avg_l2']
    transfer_l2 = transfer_errors['avg_l2']
    accuracy_ratio = transfer_l2 / fresh_l2

    print(f"\nFresh Heat PINN:    Avg L2 Error = {fresh_l2:.6f}")
    print(f"Transfer Heat PINN: Avg L2 Error = {transfer_l2:.6f}")
    print(f"\nTransfer Accuracy Ratio: {accuracy_ratio:.3f}")

    if accuracy_ratio < 1.0:
        improvement = (1 - accuracy_ratio) * 100
        print(f"✓ Transfer learning IMPROVED accuracy by {improvement:.1f}%")
        print("\nCONCLUSION: Network learned transferable mathematical structure!")
    else:
        degradation = (accuracy_ratio - 1) * 100
        print(f"✗ Transfer learning DEGRADED accuracy by {degradation:.1f}%")
        print("\nCONCLUSION: Transfer did not help. Possible causes:")
        print("  - Insufficient normalization")
        print("  - Domain mismatch")
        print("  - Network capacity issues")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    return {
        'bs_losses': bs_losses,
        'transfer_losses': transfer_losses,
        'fresh_losses': fresh_losses,
        'fresh_errors': fresh_errors,
        'transfer_errors': transfer_errors,
        'accuracy_ratio': accuracy_ratio
    }


if __name__ == '__main__':
    results = run_transfer_learning_experiment()
