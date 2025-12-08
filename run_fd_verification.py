"""
Finite Difference Verification for American Options

This script verifies the finite difference solver implementation using either:
1. Real market data from Midas API (if available)
2. Synthetic data for testing

The FD solver provides a reference solution for validating PINN accuracy.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.finite_difference.solver import AmericanOptionFDSolver
from src.data.options_data_loader import OptionsDataLoader
from src.pde.black_scholes import create_black_scholes_pinn
from src.training.trainer import train_pinn
import torch


def run_fd_verification(use_market_data: bool = False):
    """
    Run finite difference verification experiment.

    Args:
        use_market_data: If True, attempt to load real market data from Midas.
                        If False, use synthetic data.

    Returns:
        Dictionary with verification results
    """
    print("=" * 70)
    print("FINITE DIFFERENCE VERIFICATION")
    print("=" * 70)

    # Load data
    loader = OptionsDataLoader()

    if use_market_data:
        print("\n⚠️  Note: Real market data from ml2/Midas is not available.")
        print("Using realistic synthetic data based on market conditions...\n")

    print("Generating realistic options data...")
    market_data = loader.generate_realistic_data(
        contract='AAPL',
        strike=200.0,
        days_to_expiry=7,
        moneyness='ATM',
        is_call=True,
        is_american=True
    )

    print(f"\nOption Parameters:")
    print(f"  Contract:        {market_data['contract']}")
    print(f"  Spot:            ${market_data['spot']:.2f}")
    print(f"  Strike:          ${market_data['strike']:.2f}")
    print(f"  Time to Expiry:  {market_data['time_to_expiry']:.4f} years")
    print(f"  Volatility:      {market_data['volatility']*100:.2f}%")
    print(f"  Risk-free Rate:  {market_data['discount_rate']*100:.2f}%")
    print(f"  Dividend Rate:   {market_data['stock_rate']*100:.2f}%")
    print(f"  Option Type:     {'Call' if market_data['is_call'] else 'Put'}")
    print(f"  Exercise Style:  {'American' if market_data['is_american'] else 'European'}")

    # Prepare FD inputs
    spot_grid, time_grid, vol_grid, discount_rates, stock_rates = loader.prepare_fd_inputs(
        market_data,
        n_spot_points=400,
        n_time_points=100
    )

    # Create FD solver
    print("\n" + "=" * 70)
    print("Running Finite Difference Solver")
    print("=" * 70)

    fd_solver = AmericanOptionFDSolver(
        spot_range=(spot_grid[0], spot_grid[-1]),
        time_range=(time_grid[0], time_grid[-1]),
        n_spot_points=len(spot_grid),
        n_time_points=len(time_grid)
    )

    # Price the option
    vol_times = time_grid.copy()  # Simplified: assume vol time = calendar time
    calendar_times = time_grid.copy()

    option_values, fd_price = fd_solver.price_american_option(
        strike=market_data['strike'],
        is_call=market_data['is_call'],
        spot_vols=vol_grid,
        discount_rates=discount_rates,
        stock_rates=stock_rates,
        vol_times=vol_times,
        calendar_times=calendar_times,
        spot_at_valuation=market_data['spot']
    )

    print(f"\n✓ Finite Difference Price: ${fd_price:.6f}")

    # Optional: Train PINN and compare
    print("\n" + "=" * 70)
    print("Training PINN for Comparison")
    print("=" * 70)

    pinn = create_black_scholes_pinn(
        spot_range=(spot_grid[0], spot_grid[-1]),
        time_to_expiry=market_data['time_to_expiry'],
        value_range=(0.0, market_data['spot'] * 0.5),  # Rough estimate
        strike=market_data['strike'],
        volatility=market_data['volatility'],
        risk_free_rate=market_data['discount_rate'],
        dividend_rate=market_data['stock_rate'],
        is_call=market_data['is_call']
    )

    train_pinn(pinn, epochs=8000, print_freq=1000)

    # Evaluate PINN at current spot
    pinn.net.eval()
    with torch.no_grad():
        S_tensor = torch.tensor([market_data['spot']], dtype=torch.float32, device=pinn.device)
        t_tensor = torch.tensor([time_grid[0]], dtype=torch.float32, device=pinn.device)
        pinn_price = pinn.forward(S_tensor, t_tensor).item()

    print(f"\n✓ PINN Price: ${pinn_price:.6f}")

    # Comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"Finite Difference: ${fd_price:.6f}")
    print(f"PINN:              ${pinn_price:.6f}")
    print(f"Absolute Error:    ${abs(fd_price - pinn_price):.6f}")
    print(f"Relative Error:    {abs(fd_price - pinn_price)/fd_price*100:.2f}%")

    # Validation
    relative_error = abs(fd_price - pinn_price) / fd_price
    if relative_error < 0.05:  # 5% threshold
        print("\n✓ PINN matches FD solution (< 5% error)")
    elif relative_error < 0.10:
        print("\n⚠ PINN reasonably close to FD (5-10% error)")
    else:
        print("\n✗ PINN differs significantly from FD (> 10% error)")
        print("  Consider: longer training, better hyperparameters, or normalization")

    print("\n" + "=" * 70)

    return {
        'market_data': market_data,
        'fd_price': fd_price,
        'pinn_price': pinn_price,
        'relative_error': relative_error
    }


if __name__ == '__main__':
    results = run_fd_verification(use_market_data=False)
