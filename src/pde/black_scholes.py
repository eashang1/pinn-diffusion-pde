# Black-Scholes PDE: ∂V/∂t + 0.5*σ²*S²*∂²V/∂S² + (r-q)*S*∂V/∂S - r*V = 0

from ..pinn.normalized_pinn import NormalizedPINN


def create_black_scholes_pinn(
    spot_range: tuple = (0.0, 400.0),
    time_to_expiry: float = 0.02,
    value_range: tuple = (0.0, 50.0),
    strike: float = 200.0,
    volatility: float = 0.2,
    risk_free_rate: float = 0.05,
    dividend_rate: float = 0.03,
    is_call: bool = True
) -> NormalizedPINN:
    config = {
        'S_range': spot_range,
        't_range': (0.0, time_to_expiry),
        'V_range': value_range,
        'pde_type': 'black_scholes',
        'parameters': {
            'strike': strike,
            'volatility': volatility,
            'risk_free_rate': risk_free_rate,
            'dividend_rate': dividend_rate,
            'is_call': is_call
        }
    }

    return NormalizedPINN(config)


def black_scholes_analytical(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool = True
) -> float:
    import numpy as np
    from scipy.stats import norm

    if T <= 0:
        return max(S - K, 0) if is_call else max(K - S, 0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if is_call:
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return price
