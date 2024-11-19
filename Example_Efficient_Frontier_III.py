
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

def run_efficient_frontier_analysis(returns: pd.DataFrame, asset_mapping: Dict):
    """
    Run efficient frontier analysis with robust optimization using synthetic data
    and asset-specific parameters
    """
    print("\nInitializing Robust Efficient Frontier Analysis...")
    
    # Generate expected returns using Black-Litterman approach
    expected_returns_bl = generate_expected_returns(
        returns, 
        method='black_litterman',
        lookback_window=36,
        confidence=0.75
    )
    
    # Convert to DataFrame with proper shape
    expected_returns = pd.DataFrame(
        {col: expected_returns_bl[i] for i, col in enumerate(returns.columns)}, 
        index=[returns.index[-1]]  # Use last date as single row
    )
    
    # Generate asset-specific epsilons based on volatility and asset class
    base_epsilon = 0.1
    epsilon_values = generate_asset_specific_epsilon(returns, asset_mapping, base_epsilon)
    
    # Create epsilon DataFrame with proper shape
    epsilons = pd.DataFrame(
        {col: epsilon_values[i] for i, col in enumerate(returns.columns)},
        index=[returns.index[-1]]  # Use last date as single row
    )
    
    # Define asset-specific risk aversion parameters
    alpha_by_class = {
        'Developed Equities': {
            'US Large Cap': 1.0,
            'US Small Cap': 1.2,
            'International': 1.1
        },
        'Emerging Markets': {
            'Broad EM': 1.8,
            'Regional': 1.6
        },
        'Fixed Income': {
            'Government': 0.6,
            'Corporate': 0.8,
            'High Yield': 1.0
        },
        'Alternative': {
            'Real Estate': 1.3,
            'Commodities': 1.4,
            'Alternatives': 1.5
        }
    }
    
    # Create alpha DataFrame with proper shape
    alphas = pd.DataFrame(
        {col: alpha_by_class[asset_mapping[col]['class']][asset_mapping[col]['sub_class']]
         for col in returns.columns},
        index=[returns.index[-1]]  # Use last date as single row
    )
        
    # Create benchmark weights with asset class limits
    asset_class_weights = {
        'Developed Equities': 0.40,  # 40% in developed markets
        'Emerging Markets': 0.15,    # 15% in emerging markets
        'Fixed Income': 0.35,        # 35% in fixed income
        'Alternative': 0.10          # 10% in alternatives
    }
    
    # Calculate benchmark weights for individual assets
    benchmark_weights = np.zeros(len(returns.columns))
    for i, col in enumerate(returns.columns):
        asset_class = asset_mapping[col]['class']
        # Equal weight within asset class
        assets_in_class = sum(1 for c in returns.columns 
                            if asset_mapping[c]['class'] == asset_class)
        benchmark_weights[i] = asset_class_weights[asset_class] / assets_in_class
    
    try:
        # Initialize calculator with enhanced parameters
        calculator = RobustEfficientFrontier(
            returns=returns,
            expected_returns=expected_returns,
            epsilon=epsilons,
            alpha=alphas,
            omega_method='bayes',
            optimization_method=OptimizationMethod.CVXPY,
            half_life=36,
            risk_free_rate=0.02/12,
            transaction_cost=0.002
        )
        
        # Define comprehensive constraints with sub-class limits
        constraints = OptimizationConstraints(
            long_only=True,
            box_constraints={
                i: (0.0, min(0.25, 2.0 * benchmark_weights[i]))  # Cap at 25% or 2x benchmark
                for i in range(len(returns.columns))
            },
            group_constraints={
                # Asset class constraints
                'Developed_Equities': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Developed Equities'],
                    bounds=(0.3, 0.5)  # 30-50% in developed markets
                ),
                'Fixed_Income': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Fixed Income'],
                    bounds=(0.25, 0.45)  # 25-45% in fixed income
                ),
                'Emerging_Markets': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Emerging Markets'],
                    bounds=(0.05, 0.20)  # 5-20% in emerging markets
                ),
                'Alternative': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Alternative'],
                    bounds=(0.05, 0.15)  # 5-15% in alternatives
                ),
                # Sub-class constraints
                'US_Large_Cap': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['sub_class'] == 'US Large Cap'],
                    bounds=(0.15, 0.30)  # 15-30% in US Large Cap
                ),
                'Government_Bonds': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['sub_class'] == 'Government'],
                    bounds=(0.10, 0.25)  # 10-25% in Government bonds
                )
            },
            max_tracking_error=0.04,  # 4% tracking error limit
            benchmark_weights=benchmark_weights
        )
        
        print("\nComputing efficient frontier...")
        # Calculate benchmark characteristics
        benchmark_return = returns.mean() @ benchmark_weights
        benchmark_risk = np.sqrt(benchmark_weights @ returns.cov() @ benchmark_weights)
        print(f"\nBenchmark Characteristics:")
        print(f"Return: {benchmark_return*12:.2%}")  # Annualized
        print(f"Risk: {benchmark_risk*np.sqrt(12):.2%}")  # Annualized
        
        # Compute frontier with advanced parameters
        results = calculator.compute_efficient_frontier(
            n_points=25,
            epsilon_range=(
                np.minimum(epsilon_values * 0.7, 0.05),  # Lower bound
                np.maximum(epsilon_values * 1.3, 0.30)   # Upper bound
            ),
            alpha_scale_range=(0.8, 1.2),
            constraints=constraints
        )
        
        # Annualize metrics
        results['returns'] = results['returns'] * 12
        results['risks'] = results['risks'] * np.sqrt(12)
        results['sharpe_ratios'] = (results['returns'] - calculator.risk_free_rate * 12) / results['risks']
        
        # Calculate tracking errors
        tracking_errors = []
        cov_matrix = returns.cov().values
        for weights in results['weights']:
            tracking_error = np.sqrt(
                (weights - benchmark_weights).T @ 
                cov_matrix @ 
                (weights - benchmark_weights)
            ) * np.sqrt(12)
            tracking_errors.append(tracking_error)
        
        results['tracking_errors'] = np.array(tracking_errors)
        
        # Print detailed results
        print("\nFrontier Computation Complete:")
        print(f"Number of portfolios: {len(results['risks'])}")
        print(f"Risk range: {results['risks'].min():.2%} to {results['risks'].max():.2%}")
        print(f"Return range: {results['returns'].min():.2%} to {results['returns'].max():.2%}")
        print(f"Sharpe ratio range: {results['sharpe_ratios'].min():.2f} to {results['sharpe_ratios'].max():.2f}")
        print(f"Tracking error range: {min(tracking_errors):.2%} to {max(tracking_errors):.2%}")
        
        # Analyze optimal portfolio
        optimal_idx = np.argmax(results['sharpe_ratios'])
        optimal_weights = results['weights'][optimal_idx]
        
        print("\nOptimal Portfolio Composition:")
        # By asset class
        print("\nBy Asset Class:")
        for asset_class in set(asset_mapping[col]['class'] for col in returns.columns):
            class_weight = sum(
                optimal_weights[i] for i, col in enumerate(returns.columns)
                if asset_mapping[col]['class'] == asset_class
            )
            print(f"{asset_class:20s}: {class_weight:8.2%}")
            
        # By sub-class
        print("\nBy Sub-Class:")
        for asset_class in alpha_by_class:
            for sub_class in alpha_by_class[asset_class]:
                sub_class_weight = sum(
                    optimal_weights[i] for i, col in enumerate(returns.columns)
                    if asset_mapping[col]['sub_class'] == sub_class
                )
                if sub_class_weight > 0.01:  # Show only significant allocations
                    print(f"{sub_class:20s}: {sub_class_weight:8.2%}")
        
        return results
        
    except Exception as e:
        print(f"\nError in frontier computation: {str(e)}")
        raise
       
def run_complete_analysis():
    """
    Run complete efficient frontier analysis with synthetic data
    """
    # 1. Generate synthetic data
    print("Generating synthetic multi-asset data...")
    data = generate_synthetic_multi_asset_data(
        n_years=5,      # 5 years of history
        freq='M',       # Monthly frequency
        seed=42         # For reproducibility
    )
    
    # 2. Run efficient frontier analysis
    print("\nRunning efficient frontier analysis...")
    frontier_results = run_efficient_frontier_analysis(
        returns=data['returns'],
        asset_mapping=data['asset_mapping']
    )
    
    # 3. Create visualization
    print("\nCreating visualizations...")
    create_analysis_plots(frontier_results, data)
    
    return {
        'data': data,
        'frontier_results': frontier_results
    }

def create_analysis_plots(frontier_results: dict, data: dict):
    """Create comprehensive visualization of the results"""
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(2, 2)
    
    # 1. Efficient Frontier Plot
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(
        frontier_results['risks'],
        frontier_results['returns'],
        c=frontier_results['sharpe_ratios'],
        cmap='viridis',
        s=100
    )
    plt.colorbar(sc, ax=ax1, label='Sharpe Ratio')
    ax1.set_xlabel('Risk (Annualized Volatility)')
    ax1.set_ylabel('Expected Return (Annualized)')
    ax1.set_title('Robust Efficient Frontier')
    ax1.grid(True)
    
    # Format axes as percentages
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    
    # 2. Asset Allocation Evolution
    ax2 = fig.add_subplot(gs[0, 1])
    asset_classes = sorted(set(data['asset_mapping'][col]['class'] 
                             for col in data['returns'].columns))
    
    # Calculate allocations by asset class
    class_allocations = np.zeros((len(frontier_results['risks']), len(asset_classes)))
    for i, weights in enumerate(frontier_results['weights']):
        for j, asset_class in enumerate(asset_classes):
            class_allocations[i, j] = sum(
                weights[k] for k, col in enumerate(data['returns'].columns)
                if data['asset_mapping'][col]['class'] == asset_class
            )
    
    ax2.stackplot(
        frontier_results['risks'],
        class_allocations.T,
        labels=asset_classes,
        alpha=0.8
    )
    ax2.set_xlabel('Risk (Annualized Volatility)')
    ax2.set_ylabel('Allocation')
    ax2.set_title('Asset Class Allocation by Risk Level')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    
    # 3. Tracking Error Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    sc2 = ax3.scatter(
        frontier_results['tracking_errors'],
        frontier_results['returns'],
        c=frontier_results['sharpe_ratios'],
        cmap='viridis',
        s=100
    )
    plt.colorbar(sc2, ax=ax3, label='Sharpe Ratio')
    ax3.set_xlabel('Tracking Error')
    ax3.set_ylabel('Expected Return')
    ax3.set_title('Return vs Tracking Error')
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    
    # 4. Risk Contribution Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    optimal_idx = np.argmax(frontier_results['sharpe_ratios'])
    optimal_weights = frontier_results['weights'][optimal_idx]
    
    # Calculate risk contributions
    cov_matrix = data['returns'].cov().values
    total_risk = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
    marginal_risk = cov_matrix @ optimal_weights / total_risk
    risk_contributions = optimal_weights * marginal_risk
    
    # Group by asset class
    class_risk_contrib = {}
    for asset_class in asset_classes:
        class_risk_contrib[asset_class] = sum(
            risk_contributions[i] for i, col in enumerate(data['returns'].columns)
            if data['asset_mapping'][col]['class'] == asset_class
        )
    
    # Plot risk contributions
    ax4.pie(
        list(class_risk_contrib.values()),
        labels=list(class_risk_contrib.keys()),
        autopct='%1.1f%%',
        startangle=90
    )
    ax4.set_title('Risk Contribution of Optimal Portfolio')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Run the analysis
        results = run_complete_analysis()
        
        # Access components if needed
        synthetic_data = results['data']
        frontier_results = results['frontier_results']
        
        print("\nAnalysis complete! Check the visualizations for detailed results.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
