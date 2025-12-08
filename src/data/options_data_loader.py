import numpy as np
from typing import Dict, Tuple
from datetime import datetime, timedelta


class OptionsDataLoader:
    def __init__(self):
        self.market_scenarios = {
            'AAPL': {
                'name': 'Apple Inc.',
                'spot_base': 220.0,
                'volatility_base': 0.25,
                'typical_strikes': [180, 190, 200, 210, 220, 230, 240, 250],
            },
            'SPY': {
                'name': 'SPDR S&P 500 ETF',
                'spot_base': 450.0,
                'volatility_base': 0.18,
                'typical_strikes': [420, 430, 440, 450, 460, 470, 480],
            },
            'TSLA': {
                'name': 'Tesla Inc.',
                'spot_base': 250.0,
                'volatility_base': 0.55,
                'typical_strikes': [200, 220, 240, 260, 280, 300],
            },
        }

    def generate_realistic_data(
        self,
        contract: str = 'AAPL',
        strike: float = 200.0,
        days_to_expiry: int = 7,
        moneyness: str = 'ATM',
        is_call: bool = True,
        is_american: bool = True
    ) -> Dict:
        if contract in self.market_scenarios:
            scenario = self.market_scenarios[contract]
            vol_base = scenario['volatility_base']
        else:
            vol_base = 0.25

        if moneyness == 'ITM':
            spot = strike * (1.1 if is_call else 0.9)
        elif moneyness == 'OTM':
            spot = strike * (0.9 if is_call else 1.1)
        else:
            spot = strike * (1.0 + np.random.uniform(-0.02, 0.02))

        time_to_expiry = days_to_expiry / 365.0
        vol_adjustment = 1.0 + (0.1 if time_to_expiry < 0.05 else 0.0)
        volatility = vol_base * vol_adjustment * np.random.uniform(0.95, 1.05)

        risk_free_rate = 0.045 + np.random.uniform(-0.005, 0.005)

        dividend_rate = {'AAPL': 0.005, 'SPY': 0.015, 'TSLA': 0.0}.get(contract, 0.01)

        expiry_date = datetime.now() + timedelta(days=days_to_expiry)
        expiry_str = expiry_date.strftime('%Y-%m-%d')

        return {
            'contract': contract,
            'expiry': expiry_str,
            'strike': float(strike),
            'spot': float(spot),
            'volatility': float(volatility),
            'discount_rate': float(risk_free_rate),
            'stock_rate': float(dividend_rate),
            'time_to_expiry': float(time_to_expiry),
            'is_call': is_call,
            'is_american': is_american,
            'moneyness': moneyness,
            'days_to_expiry': days_to_expiry
        }

    def load_synthetic_data(
        self,
        spot: float = 220.0,
        strike: float = 200.0,
        time_to_expiry: float = 0.02,
        volatility: float = 0.2,
        risk_free_rate: float = 0.05,
        dividend_rate: float = 0.03,
        is_call: bool = True,
        is_american: bool = True
    ) -> Dict:
        days_to_expiry = int(time_to_expiry * 365)
        expiry_date = datetime.now() + timedelta(days=days_to_expiry)

        return {
            'contract': 'CUSTOM',
            'expiry': expiry_date.strftime('%Y-%m-%d'),
            'strike': float(strike),
            'spot': float(spot),
            'volatility': float(volatility),
            'discount_rate': float(risk_free_rate),
            'stock_rate': float(dividend_rate),
            'time_to_expiry': float(time_to_expiry),
            'is_call': is_call,
            'is_american': is_american,
            'days_to_expiry': days_to_expiry
        }

    def prepare_fd_inputs(
        self,
        market_data: Dict,
        n_spot_points: int = 400,
        n_time_points: int = 100,
        spot_multiplier: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        spot = market_data['spot']
        T = market_data['time_to_expiry']
        vol = market_data['volatility']
        r = market_data['discount_rate']
        q = market_data['stock_rate']

        # Create grids
        spot_grid = np.linspace(0, spot * spot_multiplier, n_spot_points)
        time_grid = np.linspace(0, T, n_time_points)

        # Create constant vol surface
        vol_grid = np.full((n_time_points, n_spot_points), vol)

        # Constant rates
        discount_rates = np.full(n_time_points, r)
        stock_rates = np.full(n_time_points, q)

        return spot_grid, time_grid, vol_grid, discount_rates, stock_rates
