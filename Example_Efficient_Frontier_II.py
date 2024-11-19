# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:08:34 2024

@author: Dennis Jung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PortOpt import *
from typing import Dict
import os


def get_multi_asset_test_data(
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


def analyze_universe(data: Dict):
    """Analyze the multi-asset universe"""
    returns = data['returns']
    asset_mapping = data['asset_mapping']
    
    # Calculate key statistics
    stats = pd.DataFrame(columns=['Annual Return', 'Annual Vol', 'Sharpe Ratio', 'Max Drawdown', 
                                'Skew', 'Kurtosis', 'Best Month', 'Worst Month'])
    
    for symbol in returns.columns:
        asset_info = asset_mapping[symbol]
        r = returns[symbol]
        
        # Calculate statistics
        annual_return = (1 + r.mean()) ** 12 - 1
        annual_vol = r.std() * np.sqrt(12)
        sharpe = (annual_return - data['risk_free_rate'] * 12) / annual_vol
        
        # Calculate drawdown
        cum_returns = (1 + r).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Higher moments and extremes
        skew = r.skew()
        kurt = r.kurtosis()
        best_month = r.max()
        worst_month = r.min()
        
        stats.loc[symbol] = [annual_return, annual_vol, sharpe, max_drawdown,
                           skew, kurt, best_month, worst_month]
    
    # Add asset class information
    stats['Asset Class'] = [asset_mapping[symbol]['class'] for symbol in stats.index]
    stats['Sub Class'] = [asset_mapping[symbol]['sub_class'] for symbol in stats.index]
    
    # Create visualization dashboard
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2)
    
    # 1. Risk-Return Scatter by Asset Class
    ax1 = fig.add_subplot(gs[0, 0])
    for asset_class in stats['Asset Class'].unique():
        mask = stats['Asset Class'] == asset_class
        ax1.scatter(
            stats.loc[mask, 'Annual Vol'],
            stats.loc[mask, 'Annual Return'],
            label=asset_class,
            alpha=0.7,
            s=100
        )
    ax1.set_xlabel('Annual Volatility')
    ax1.set_ylabel('Annual Return')
    ax1.set_title('Risk-Return by Asset Class')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Correlation Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(data['correlation'], cmap='RdYlBu_r', aspect='auto')
    plt.colorbar(im, ax=ax2)
    ax2.set_xticks(np.arange(len(returns.columns)))
    ax2.set_yticks(np.arange(len(returns.columns)))
    ax2.set_xticklabels(returns.columns, rotation=45, ha='right')
    ax2.set_yticklabels(returns.columns)
    ax2.set_title('Cross-Asset Correlation Matrix')
    
    # 3. Asset Class Performance Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    class_returns = stats.groupby('Asset Class')['Annual Return'].mean().sort_values()
    ax3.barh(range(len(class_returns)), class_returns.values)
    ax3.set_yticks(range(len(class_returns)))
    ax3.set_yticklabels(class_returns.index)
    ax3.set_title('Average Annual Return by Asset Class')
    ax3.set_xlabel('Annual Return')
    
    # 4. Risk Metrics Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    risk_metrics = stats.groupby('Asset Class').agg({
        'Annual Vol': 'mean',
        'Max Drawdown': 'mean',
        'Skew': 'mean'
    }).round(2)
    x = np.arange(len(risk_metrics.index))
    width = 0.25
    
    ax4.bar(x - width, risk_metrics['Annual Vol'], width, label='Annual Vol')
    ax4.bar(x, risk_metrics['Max Drawdown'], width, label='Max Drawdown')
    ax4.bar(x + width, risk_metrics['Skew'], width, label='Skew')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(risk_metrics.index, rotation=45)
    ax4.set_title('Risk Metrics by Asset Class')
    ax4.legend()
    
    # 5. Distribution Analysis
    ax5 = fig.add_subplot(gs[2, 0])
    for asset_class in stats['Asset Class'].unique():
        mask = stats['Asset Class'] == asset_class
        ax5.hist(
            returns[returns.columns[mask]].mean(),
            alpha=0.5,
            label=asset_class,
            bins=20
        )
    ax5.set_title('Return Distribution by Asset Class')
    ax5.set_xlabel('Monthly Return')
    ax5.legend()
    
    # 6. Time Series of Cumulative Returns
    ax6 = fig.add_subplot(gs[2, 1])
    cum_returns = (1 + returns).cumprod()
    for asset_class in stats['Asset Class'].unique():
        mask = stats['Asset Class'] == asset_class
        class_return = cum_returns[cum_returns.columns[mask]].mean(axis=1)
        ax6.plot(class_return.index, class_return.values, label=asset_class)
    ax6.set_title('Cumulative Returns by Asset Class')
    ax6.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\nDetailed Statistics by Asset Class:")
    print("-" * 80)
    summary_stats = stats.groupby('Asset Class').agg({
        'Annual Return': ['mean', 'std'],
        'Annual Vol': ['mean', 'std'],
        'Sharpe Ratio': ['mean', 'std'],
        'Max Drawdown': ['mean', 'std'],
        'Skew': 'mean',
        'Kurtosis': 'mean'
    }).round(3)
    
    print(summary_stats)
    
    return stats

def analyze_efficient_frontier_results(frontier_results: Dict[str, np.ndarray],
                                    returns: pd.DataFrame,
                                    asset_mapping: Dict,
                                    epsilon: float):
    """Create visualization of efficient frontier results with tracking error analysis"""
    
    output_dir = 'Output_Efficient_Frontier'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated output directory: {output_dir}")
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 8))
    gs = plt.GridSpec(1, 3)
    
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
    ax1.set_xlabel('Risk (Volatility)')
    ax1.set_ylabel('Expected Return')
    ax1.set_title(f'Robust Efficient Frontier (ε={epsilon:.2f})')
    ax1.grid(True, alpha=0.3)
    
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    
    # 2. Asset Allocation Evolution
    ax2 = fig.add_subplot(gs[0, 1])
    asset_classes = sorted(set(asset_mapping[col]['class'] for col in returns.columns))
    allocations = np.zeros((len(frontier_results['risks']), len(asset_classes)))
    
    for i, weights in enumerate(frontier_results['weights']):
        for j, asset_class in enumerate(asset_classes):
            allocations[i, j] = sum(
                weights[k] for k, col in enumerate(returns.columns)
                if asset_mapping[col]['class'] == asset_class
            )
    
    ax2.stackplot(
        frontier_results['risks'],
        allocations.T,
        labels=asset_classes,
        alpha=0.8
    )
    ax2.set_xlabel('Risk (Volatility)')
    ax2.set_ylabel('Allocation')
    ax2.set_title('Asset Class Allocation Evolution')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    
    # 3. Tracking Error Analysis
    ax3 = fig.add_subplot(gs[0, 2])
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
    ax3.grid(True, alpha=0.3)
    
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'efficient_frontier_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and save analysis results (rest of the function remains unchanged)
    optimal_idx = np.argmax(frontier_results['sharpe_ratios'])
    optimal_weights = frontier_results['weights'][optimal_idx]
    
    optimal_stats = {
        'Expected Return': frontier_results['returns'][optimal_idx],
        'Risk': frontier_results['risks'][optimal_idx],
        'Sharpe Ratio': frontier_results['sharpe_ratios'][optimal_idx],
        'Tracking Error': frontier_results['tracking_errors'][optimal_idx]
    }
    
    print("\nOptimal Portfolio Characteristics:")
    optimal_stats_df = pd.DataFrame.from_dict(optimal_stats, orient='index', columns=['Value'])
    optimal_stats_df.to_excel(os.path.join(output_dir, 'optimal_portfolio_stats.xlsx'))
    print(f"Expected Return: {optimal_stats['Expected Return']:.2%}")
    print(f"Risk: {optimal_stats['Risk']:.2%}")
    print(f"Sharpe Ratio: {optimal_stats['Sharpe Ratio']:.2f}")
    print(f"Tracking Error: {optimal_stats['Tracking Error']:.2%}")
    
    class_alloc = {}
    for i, col in enumerate(returns.columns):
        asset_class = asset_mapping[col]['class']
        class_alloc[asset_class] = class_alloc.get(asset_class, 0) + optimal_weights[i]
    
    print("\nOptimal Asset Class Allocation:")
    class_alloc_df = pd.DataFrame.from_dict(class_alloc, orient='index', columns=['Allocation'])
    class_alloc_df.to_excel(os.path.join(output_dir, 'optimal_class_allocation.xlsx'))
    for asset_class, alloc in sorted(class_alloc.items()):
        print(f"{asset_class:20s}: {alloc:8.2%}")
    
    allocations_df = pd.DataFrame(
        allocations,
        columns=asset_classes,
        index=[f'Portfolio_{i}' for i in range(len(frontier_results['risks']))]
    )
    allocations_df.to_excel(os.path.join(output_dir, 'allocation_evolution.xlsx'))
    
    frontier_df = pd.DataFrame({
        'Risk': frontier_results['risks'],
        'Return': frontier_results['returns'],
        'Sharpe_Ratio': frontier_results['sharpe_ratios'],
        'Tracking_Error': frontier_results['tracking_errors']
    })
    frontier_df.to_excel(os.path.join(output_dir, 'frontier_points.xlsx'))
    
    weights_df = pd.DataFrame(
        frontier_results['weights'],
        columns=returns.columns,
        index=[f'Portfolio_{i}' for i in range(len(frontier_results['risks']))]
    )
    weights_df.to_excel(os.path.join(output_dir, 'portfolio_weights.xlsx'))
    
    summary_stats = pd.DataFrame({
        'Metric': ['Minimum', 'Maximum', 'Average'],
        'Return': [
            f"{frontier_results['returns'].min():.2%}",
            f"{frontier_results['returns'].max():.2%}",
            f"{frontier_results['returns'].mean():.2%}"
        ],
        'Risk': [
            f"{frontier_results['risks'].min():.2%}",
            f"{frontier_results['risks'].max():.2%}",
            f"{frontier_results['risks'].mean():.2%}"
        ],
        'Sharpe Ratio': [
            f"{frontier_results['sharpe_ratios'].min():.2f}",
            f"{frontier_results['sharpe_ratios'].max():.2f}",
            f"{frontier_results['sharpe_ratios'].mean():.2f}"
        ],
        'Tracking Error': [
            f"{frontier_results['tracking_errors'].min():.2%}",
            f"{frontier_results['tracking_errors'].max():.2%}",
            f"{frontier_results['tracking_errors'].mean():.2%}"
        ]
    }).set_index('Metric')
    
    summary_stats.to_excel(os.path.join(output_dir, 'frontier_summary_stats.xlsx'))
    
    return {
        'optimal_weights': optimal_weights,
        'class_allocation': class_alloc,
        'allocations_evolution': allocations_df,
        'frontier_points': frontier_df,
        'portfolio_weights': weights_df,
        'summary_stats': summary_stats
    }
    
def run_complete_analysis():
    """Run complete efficient frontier analysis with visualizations"""
    
    # Get test data
    print("Fetching market data...")
    data = get_multi_asset_test_data()
    
    # Analyze universe
    print("\nAnalyzing investment universe...")
    universe_stats = analyze_universe(data)
    
    # Run efficient frontier analysis
    print("\nComputing efficient frontier...")
    frontier_results = run_efficient_frontier_analysis(data['returns'], data['asset_mapping'])
    
    # Analyze frontier results
    print("\nAnalyzing efficient frontier results...")
    analysis_results = analyze_efficient_frontier_results(
        frontier_results,
        data['returns'],
        data['asset_mapping'],
        epsilon=0.1
    )
    
    return {
        'universe_stats': universe_stats,
        'frontier_results': frontier_results,
        'analysis_results': analysis_results
    }

def run_efficient_frontier_analysis(returns: pd.DataFrame, asset_mapping: Dict):
    """Run efficient frontier analysis with robust optimization, proper scaling, and tracking error constraint"""
    print("\nInitializing Robust Efficient Frontier Analysis...")
    
    # Define asset-specific risk aversion parameters
    alpha_by_class = {
        'Developed Equities': 1.2,    
        'Emerging Markets': 1.8,      
        'Fixed Income': 0.8,          
        'Alternative': 1.5            
    }
    
    # Create alpha vector
    alphas = np.array([
        alpha_by_class[asset_mapping[col]['class']] 
        for col in returns.columns
    ])
    
    # Create equal-weighted benchmark
    n_assets = len(returns.columns)
    benchmark_weights = np.ones(n_assets) / n_assets
    
    try:
        # Initialize calculator with adjusted parameters
        calculator = RobustEfficientFrontier(
            optimization_method=OptimizationMethod.CVXPY,
            returns=returns,
            epsilon=0.1,              
            alpha=alphas,             
            half_life=36,             
            risk_free_rate=0.02/12,   # Convert annual rate to monthly
            transaction_cost=0.002     
        )
        
        # Define constraints with tracking error
        constraints = OptimizationConstraints(
            long_only=True,
            box_constraints={
                i: (0.0, 0.50) for i in range(len(returns.columns))  # Max 25% per asset
            },
            group_constraints={
                'Developed Equities': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Developed Equities'],
                    bounds=(0., 0.5)  # 1-50% in developed markets
                ),
                'Fixed Income': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Fixed Income'],
                    bounds=(0., 0.4)  # 1-40% in fixed income
                ),
                'Emerging Markets': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Emerging Markets'],
                    bounds=(0., 0.5)  # 1-25% in emerging markets
                ),
                'Alternative': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Alternative'],
                    bounds=(0., 0.4)  # 1-20% in alternatives
                )
            },
            # Add tracking error constraint
            max_tracking_error=0.05,  # 5% tracking error
            benchmark_weights=benchmark_weights
        )
        
        print("\nComputing efficient frontier...")
        # Calculate benchmark characteristics
        benchmark_return = returns.mean() @ benchmark_weights
        benchmark_risk = np.sqrt(benchmark_weights @ returns.cov() @ benchmark_weights)
        print(f"\nBenchmark Characteristics:")
        print(f"Return: {benchmark_return*12:.2%}")  # Annualized
        print(f"Risk: {benchmark_risk*np.sqrt(12):.2%}")  # Annualized
        
        # Compute frontier with tracking error constraint
        results = calculator.compute_efficient_frontier(
            n_points=15,             
            epsilon_range=(0.05, 0.1),
            alpha_scale_range=(0.8, 1.2),
            constraints=constraints
        )
        
        # Annualize returns and risks for Sharpe ratio calculation
        results['returns'] = results['returns'] * 12  # Annualize returns
        results['risks'] = results['risks'] * np.sqrt(12)  # Annualize risks
        
        # Recalculate Sharpe ratios with annualized values
        rf_rate = calculator.risk_free_rate * 12  # Annualize risk-free rate
        results['sharpe_ratios'] = (results['returns'] - rf_rate) / results['risks']
        
        # Calculate tracking errors for each portfolio
        tracking_errors = []
        cov_matrix = returns.cov().values
        for weights in results['weights']:
            tracking_error = np.sqrt(
                (weights - benchmark_weights).T @ 
                cov_matrix @ 
                (weights - benchmark_weights)
            ) * np.sqrt(12)  # Annualize
            tracking_errors.append(tracking_error)
        
        results['tracking_errors'] = np.array(tracking_errors)
        
        print("\nFrontier Computation Complete:")
        print(f"Number of portfolios: {len(results['risks'])}")
        print(f"Risk range: {results['risks'].min():.4%} to {results['risks'].max():.4%}")
        print(f"Return range: {results['returns'].min():.4%} to {results['returns'].max():.4%}")
        print(f"Sharpe ratio range: {results['sharpe_ratios'].min():.2f} to {results['sharpe_ratios'].max():.2f}")
        print(f"Tracking error range: {min(tracking_errors):.2%} to {max(tracking_errors):.2%}")
        
        return results
        
    except Exception as e:
        print(f"\nError in frontier computation: {str(e)}")
        raise

if __name__ == "__main__":
    results = run_complete_analysis()
