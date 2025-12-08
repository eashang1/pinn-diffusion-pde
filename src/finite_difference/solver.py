import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Optional


def tridiagonal_matrix_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    # Solve tridiagonal system using Thomas algorithm
    n = len(d)
    x = np.zeros(n)
    work_array = np.zeros(n)

    temp = 1.0 / b[0]
    x[0] = d[0] * temp
    for i in range(1, n):
        work_array[i] = c[i-1] * temp
        temp = 1.0 / (b[i] - a[i] * work_array[i])
        x[i] = (d[i] - a[i] * x[i-1]) * temp

    for i in range(n-2, -1, -1):
        x[i] -= work_array[i+1] * x[i+1]

    return x


class AmericanOptionFDSolver:
    def __init__(
        self,
        spot_range: Tuple[float, float],
        time_range: Tuple[float, float],
        n_spot_points: int = 400,
        n_time_points: int = 100,
        min_variance: float = 1e-9
    ):
        self.min_var = min_variance
        self.spot_pts = np.linspace(spot_range[0], spot_range[1], n_spot_points)
        self.time_pts = np.linspace(time_range[0], time_range[1], n_time_points)
        self.n_spot = len(self.spot_pts)
        self.n_time = len(self.time_pts)

        self.ds = np.zeros(self.n_spot)
        self.ds[0] = self.spot_pts[1] - self.spot_pts[0]
        self.ds[-1] = self.spot_pts[-1] - self.spot_pts[-2]
        for j in range(1, self.n_spot - 1):
            self.ds[j] = 0.5 * (self.spot_pts[j+1] - self.spot_pts[j-1])

    def price_american_option(
        self,
        strike: float,
        is_call: bool,
        spot_vols: np.ndarray,
        discount_rates: np.ndarray,
        stock_rates: np.ndarray,
        vol_times: np.ndarray,
        calendar_times: np.ndarray,
        dividend_amounts: Optional[np.ndarray] = None,
        spot_at_valuation: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[float]]:
        if dividend_amounts is None:
            dividend_amounts = np.zeros(self.n_time)

        option_values = np.zeros(self.n_spot)
        a, b, c = np.zeros(self.n_spot), np.zeros(self.n_spot), np.zeros(self.n_spot)

        fwdpx = np.zeros((self.n_time, self.n_spot))
        for i in range(self.n_time):
            fwdpx[i, :] = self.spot_pts * np.exp(stock_rates[i] * calendar_times[i])

        option_values = np.maximum((fwdpx[-1, :] - strike) if is_call else (strike - fwdpx[-1, :]), 0.0)

        pvdivs = 0.0
        for i in range(self.n_time - 1, 0, -1):
            cal_dt = self.time_pts[i] - self.time_pts[i-1]
            vol_dt = vol_times[i] - vol_times[i-1]

            fwd_factor = np.exp(stock_rates[i] * cal_dt)
            pvdivs = (pvdivs + dividend_amounts[i]) / fwd_factor

            for j in range(self.n_spot):
                spot_var = max(self.min_var, self.spot_pts[j]**2 * spot_vols[i][j]**2)
                kstar = spot_var * 0.5 * vol_dt / (self.ds[j]**2)
                a[j], b[j], c[j] = -kstar, 1.0 + 2.0 * kstar, -kstar

            temp_val = tridiagonal_matrix_solve(a, b, c, option_values)
            discount_factor = np.exp(-discount_rates[i] * cal_dt)

            for j in range(self.n_spot):
                intrinsic = max((fwdpx[i-1, j] + pvdivs - strike) if is_call
                               else (strike - fwdpx[i-1, j] - pvdivs), 0.0)
                option_values[j] = max(intrinsic, discount_factor * temp_val[j])

        if spot_at_valuation is not None:
            interp_func = interp1d(self.spot_pts, option_values, kind='linear', fill_value='extrapolate')
            return option_values, float(interp_func(spot_at_valuation))

        return option_values, None
