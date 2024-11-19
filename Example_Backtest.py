# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:24:42 2024

@author: Dennis Jung
"""
###############################################################################
##### Example Backtest 
###############################################################################

from PortOpt import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict
import os

def generate_synthetic_multi_asset_data(
    n_years: int = 5,
    freq: str = 'M',
    seed: int = 42
) -> Dict:
    """
    Generate synthetic multi-asset universe data with improved numerical stability
    """
    np.random.seed(seed)
    
    # Define multi-asset universe structure (same as before)
    assets = {
        'Developed Equities': {
            'US Large Cap': ['US_LC1', 'US_LC2', 'US_LC3'],
            'US Small Cap': ['US_SC1', 'US_SC2'],
            'International': ['INTL1', 'INTL2', 'INTL3']
        },
        'Emerging Markets': {
            'Broad EM': ['EM1', 'EM2'],
            'Regional': ['ASIA1', 'LATAM1']
        },
        'Fixed Income': {
            'Government': ['GOVT1', 'GOVT2', 'GOVT3'],
            'Corporate': ['CORP1', 'CORP2'],
            'High Yield': ['HY1', 'HY2']
        },
        'Alternative': {
            'Real Estate': ['REIT1', 'REIT2'],
            'Commodities': ['COMM1', 'COMM2'],
            'Alternatives': ['ALT1']
        }
    }
    
    # Extract symbols and create mapping
    all_symbols = []
    asset_mapping = {}
    for asset_class, sub_classes in assets.items():
        for sub_class, symbols in sub_classes.items():
            for symbol in symbols:
                all_symbols.append(symbol)
                asset_mapping[symbol] = {
                    'class': asset_class,
                    'sub_class': sub_class
                }
    
    # Set up time index
    if freq == 'M':
        periods = n_years * 12
        annualization = 12
    else:  # Weekly
        periods = n_years * 52
        annualization = 52
        
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=n_years*365),
        periods=periods,
        freq=freq
    )
    
    # Define more conservative return parameters
    class_params = {
        'Developed Equities': {
            'mean': 0.06/annualization,  # 6% annual return
            'vol': 0.12/np.sqrt(annualization),
            'beta_market': 1.0
        },
        'Emerging Markets': {
            'mean': 0.08/annualization,  # 8% annual return
            'vol': 0.16/np.sqrt(annualization),
            'beta_market': 1.2
        },
        'Fixed Income': {
            'mean': 0.03/annualization,  # 3% annual return
            'vol': 0.04/np.sqrt(annualization),
            'beta_market': 0.2
        },
        'Alternative': {
            'mean': 0.05/annualization,  # 5% annual return
            'vol': 0.10/np.sqrt(annualization),
            'beta_market': 0.5
        }
    }
    
    # Generate market factor with more stable parameters
    market_vol = 0.12/np.sqrt(annualization)  # 12% annual volatility
    market_return = np.random.normal(
        0.05/annualization,  # 5% annual return
        market_vol,
        periods
    )
    
    # Initialize returns matrix
    n_assets = len(all_symbols)
    returns_matrix = np.zeros((periods, n_assets))
    
    # Generate common factors for each asset class
    class_factors = {}
    for asset_class in class_params.keys():
        class_factors[asset_class] = np.random.normal(0, 0.5 * market_vol, periods)
    
    # Generate returns with improved numerical stability
    for i, symbol in enumerate(all_symbols):
        asset_class = asset_mapping[symbol]['class']
        params = class_params[asset_class]
        
        # Combine market, class, and idiosyncratic factors
        systematic_return = (
            params['beta_market'] * market_return +
            0.3 * class_factors[asset_class]  # Class-specific factor
        )
        
        # Generate stable idiosyncratic returns
        idio_vol = params['vol'] * 0.4  # Reduce idiosyncratic component
        idio_returns = np.random.normal(0, idio_vol, periods)
        
        # Combine components with scaling
        returns_matrix[:, i] = (
            params['mean'] +  # Deterministic drift
            systematic_return +  # Systematic component
            idio_returns  # Idiosyncratic component
        )
        
        # Apply winsorization to extreme values
        returns_matrix[:, i] = np.clip(
            returns_matrix[:, i],
            np.percentile(returns_matrix[:, i], 1),
            np.percentile(returns_matrix[:, i], 99)
        )
    
    # Create returns DataFrame
    returns = pd.DataFrame(
        returns_matrix,
        index=dates,
        columns=all_symbols
    )
    
    # Generate prices with numerical stability checks
    prices = pd.DataFrame(index=dates, columns=all_symbols)
    for col in all_symbols:
        # Initialize at 100 and accumulate returns
        price_series = 100 * (1 + returns[col]).cumprod()
        # Ensure no zeros or infinities
        price_series = np.maximum(price_series, 1e-8)
        prices[col] = price_series
    
    # Set conservative risk-free rate
    rf_rate = 0.01 / annualization  # 1% annual
    
    # Calculate correlation matrix with stability improvements
    correlation = returns.corr()
    
    # Clean up correlation matrix
    eigenvals, eigenvecs = np.linalg.eigh(correlation)
    min_eigenval = np.min(eigenvals)
    if min_eigenval < 0:
        # Add small positive constant to diagonal
        correlation = correlation + (-min_eigenval + 1e-8) * np.eye(n_assets)
    
    # Ensure correlation matrix is well-conditioned
    condition_number = np.linalg.cond(correlation)
    if condition_number > 1e10:  # If poorly conditioned
        # Apply shrinkage
        shrinkage_factor = 0.1
        target = np.eye(n_assets)
        correlation = (1 - shrinkage_factor) * correlation + shrinkage_factor * target
    
    return {
        'returns': returns,
        'asset_mapping': asset_mapping,
        'correlation': correlation,
        'risk_free_rate': rf_rate,
        'prices': prices
    }

def generate_expected_returns(
    returns: pd.DataFrame,
    method: str = 'black_litterman',
    lookback_window: int = 36,
    confidence: float = 0.75
) -> np.ndarray:
    """
    Generate expected returns using multiple methods
    
    Args:
        returns: Historical returns
        method: 'historical', 'black_litterman', 'combined'
        lookback_window: Historical window for estimation
        confidence: Confidence level for estimates
    """
    if method == 'historical':
        return returns.rolling(lookback_window).mean().iloc[-1].values
        
    elif method == 'black_litterman':
        # Simple Black-Litterman implementation
        historical_mean = returns.mean()
        market_cap_weights = np.ones(len(returns.columns)) / len(returns.columns)
        risk_aversion = 2.5
        tau = 0.025
        
        # Prior returns (equilibrium returns)
        sigma = returns.cov()
        pi = risk_aversion * sigma @ market_cap_weights
        
        # Views
        P = np.eye(len(returns.columns))
        Q = historical_mean.values
        omega = tau * sigma
        
        # Posterior estimate
        inv_sigma = np.linalg.inv(sigma)
        post_sigma = np.linalg.inv(inv_sigma + P.T @ np.linalg.inv(omega) @ P)
        post_mean = post_sigma @ (inv_sigma @ pi + P.T @ np.linalg.inv(omega) @ Q)
        
        return post_mean
        
    else:  # combined
        hist_mean = returns.rolling(lookback_window).mean().iloc[-1].values
        bl_mean = generate_expected_returns(returns, 'black_litterman', lookback_window)
        return confidence * hist_mean + (1 - confidence) * bl_mean

def generate_asset_specific_epsilon(
    returns: pd.DataFrame,
    asset_mapping: Dict,
    base_epsilon: float = 0.1
) -> np.ndarray:
    """Generate asset-specific uncertainty parameters"""
    vols = returns.std()
    rel_vols = vols / vols.mean()
    
    # Adjust epsilon based on asset class characteristics
    class_factors = {
        'Developed Equities': 0.8,
        'Emerging Markets': 1.2,
        'Fixed Income': 0.6,
        'Alternative': 1.0
    }
    
    epsilons = np.zeros(len(returns.columns))
    for i, col in enumerate(returns.columns):
        asset_class = asset_mapping[col]['class']
        class_factor = class_factors[asset_class]
        vol_factor = rel_vols[col]
        epsilons[i] = base_epsilon * class_factor * vol_factor
        
    return np.clip(epsilons, 0.05, 0.3)

"""
Test examples for RobustBacktestOptimizer with different optimization scenarios
"""

import numpy as np 
import pandas as pd
from typing import Dict
from datetime import datetime, timedelta

def test_minimum_variance_out_of_sample():
    """Tests minimum variance optimization with out-of-sample expected returns"""
    data = generate_synthetic_multi_asset_data(n_years=5)
    returns = data['returns']
    asset_mapping = data['asset_mapping']
    
    # Generate expected returns DataFrame
    base_expected = pd.Series(index=returns.columns)
    for col in returns.columns:
        asset_class = asset_mapping[col]['class']
        if 'Equities' in asset_class:
            base_expected[col] = 0.06
        elif 'Fixed Income' in asset_class:
            base_expected[col] = 0.03
        elif 'Alternative' in asset_class:
            base_expected[col] = 0.05
        else:
            base_expected[col] = 0.04
            
    expected_returns = pd.DataFrame(
        np.tile(base_expected.values, (len(returns), 1)),
        index=returns.index,
        columns=returns.columns
    )
    
    # Create epsilon and alpha DataFrames
    base_epsilon = pd.Series(np.linspace(0.05, 0.15, len(returns.columns)), index=returns.columns)
    base_alpha = pd.Series(np.linspace(0.8, 1.2, len(returns.columns)), index=returns.columns)
    
    epsilon = pd.DataFrame(
        np.tile(base_epsilon.values, (len(returns), 1)),
        index=returns.index,
        columns=returns.columns
    )
    
    alpha = pd.DataFrame(
        np.tile(base_alpha.values, (len(returns), 1)),
        index=returns.index,
        columns=returns.columns
    )
    
    optimizer = RobustBacktestOptimizer(
        returns=returns,
        expected_returns=expected_returns,
        epsilon=epsilon,
        alpha=alpha,
        lookback_window=36,
        rebalance_frequency=3,
        out_of_sample=True
    )
    
    constraints = OptimizationConstraints(long_only=True)
    return optimizer.run_backtest(
        objective=ObjectiveFunction.MINIMUM_VARIANCE,
        constraints=constraints
    )

def test_maximum_sharpe_with_risk_target():
    """Tests maximum Sharpe optimization with risk target and group constraints"""
    data = generate_synthetic_multi_asset_data(n_years=5)
    returns = data['returns']
    asset_mapping = data['asset_mapping']
    
    # Generate expected returns DataFrame
    base_expected = pd.Series(index=returns.columns)
    for col in returns.columns:
        asset_class = asset_mapping[col]['class']
        if 'Equities' in asset_class:
            base_expected[col] = 0.08
        elif 'Fixed Income' in asset_class:
            base_expected[col] = 0.04
        elif 'Alternative' in asset_class:
            base_expected[col] = 0.06
        else:
            base_expected[col] = 0.05
    
    expected_returns = pd.DataFrame(
        np.tile(base_expected.values, (len(returns), 1)),
        index=returns.index,
        columns=returns.columns
    )
    
    # Create constant epsilon and alpha DataFrames
    epsilon = pd.DataFrame(0.1, index=returns.index, columns=returns.columns)
    alpha = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    
    # Group constraints
    group_constraints = {
        'Equities': GroupConstraint(
            assets=[i for i, col in enumerate(returns.columns) 
                   if 'Equities' in asset_mapping[col]['class']],
            bounds=(0.3, 0.6)
        ),
        'Fixed Income': GroupConstraint(
            assets=[i for i, col in enumerate(returns.columns) 
                   if 'Fixed Income' in asset_mapping[col]['class']],
            bounds=(0.2, 0.4)
        )
    }
    
    optimizer = RobustBacktestOptimizer(
        returns=returns,
        expected_returns=expected_returns,
        epsilon=epsilon,
        alpha=alpha,
        lookback_window=36,
        out_of_sample=True
    )
    
    constraints = OptimizationConstraints(
        long_only=True,
        target_risk=0.10,
        group_constraints=group_constraints
    )
    
    return optimizer.run_backtest(
        objective=ObjectiveFunction.MAXIMUM_SHARPE,
        constraints=constraints
    )

def run_all_tests():
    """Run all test cases and compile results"""
    results = {
        'minimum_variance': test_minimum_variance_out_of_sample(),
        'maximum_sharpe': test_maximum_sharpe_with_risk_target(),
    }
    
    return results

if __name__ == "__main__":
    results = run_all_tests()
    for test_name, test_results in results.items():
        print(f"\n{test_name} Results:")
        print(test_results['backtest_metrics'])