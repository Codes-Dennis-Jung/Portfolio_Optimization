import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import norm
from PortOpt import *
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

def get_multi_asset_test_data():
    """Get multi-asset universe data for testing"""
    # Define multi-asset universe
    assets = {
        'Developed Equities': {
            'US Large Cap': ['SPY', 'QQQ', 'IWB'],  # S&P 500, Nasdaq, Russell 1000
            'US Small Cap': ['IWM', 'IJR'],         # Russell 2000, S&P 600
            'International': ['EFA', 'VEA', 'IEFA']  # EAFE Markets
        },
        'Emerging Markets': {
            'Broad EM': ['EEM', 'VWO'],  # Emerging Markets
            'Regional': ['FXI', 'EWZ']    # China, Brazil
        },
        'Fixed Income': {
            'Government': ['IEF', 'TLT', 'GOVT'],  # 7-10Y, 20Y+, All Treasury
            'Corporate': ['LQD', 'VCIT'],          # Investment Grade, Intermediate Corp
            'High Yield': ['HYG', 'JNK']           # High Yield Corporate
        },
        'Alternative': {
            'Real Estate': ['VNQ', 'REM'],     # REITs
            'Commodities': ['GLD', 'USO'],     # Gold, Oil
            'Alternatives': ['QAI']            # Multi-Strategy
        }
    }
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years of data
    
    # Download data
    prices = pd.DataFrame()
    asset_mapping = {}
    failed_downloads = []
    
    print("Downloading multi-asset data...")
    for asset_class, sub_classes in assets.items():
        for sub_class, symbols in sub_classes.items():
            for symbol in symbols:
                try:
                    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if len(data) > 0:
                        prices[symbol] = data['Adj Close']
                        asset_mapping[symbol] = {
                            'class': asset_class, 
                            'sub_class': sub_class
                        }
                    else:
                        failed_downloads.append(symbol)
                except Exception as e:
                    print(f"Failed to download {symbol}: {e}")
                    failed_downloads.append(symbol)
    
    if failed_downloads:
        print(f"Warning: Failed to download data for: {', '.join(failed_downloads)}")
    
    # Resample to monthly frequency and calculate returns
    prices = prices.resample('M').last()
    returns = prices.pct_change().dropna()
    
    # Handle missing data and outliers
    returns = returns.fillna(method='ffill').fillna(method='bfill')
    
    # Winsorize extreme outliers
    for col in returns.columns:
        percentiles = returns[col].quantile([0.01, 0.99])
        returns[col] = returns[col].clip(percentiles[0.01], percentiles[0.99])
    
    # Get risk-free rate data
    try:
        rf_data = yf.download('^IRX', start=start_date, end=end_date, progress=False)['Adj Close'] / 100 / 12
        rf_rate = rf_data.mean()
    except:
        print("Warning: Using default risk-free rate")
        rf_rate = 0.02 / 12  # Default monthly risk-free rate
    
    return {
        'returns': returns,
        'asset_mapping': asset_mapping,
        'correlation': returns.corr(),
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
    plt.style.use('seaborn')
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
    sns.heatmap(data['correlation'], cmap='RdYlBu_r', center=0, ax=ax2)
    ax2.set_title('Cross-Asset Correlation Matrix')
    
    # 3. Asset Class Performance Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    class_returns = stats.groupby('Asset Class')['Annual Return'].mean().sort_values()
    class_returns.plot(kind='barh', ax=ax3, color='darkblue')
    ax3.set_title('Average Annual Return by Asset Class')
    ax3.set_xlabel('Annual Return')
    
    # 4. Risk Metrics Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    risk_metrics = stats.groupby('Asset Class').agg({
        'Annual Vol': 'mean',
        'Max Drawdown': 'mean',
        'Skew': 'mean'
    }).round(2)
    risk_metrics.plot(kind='bar', ax=ax4)
    ax4.set_title('Risk Metrics by Asset Class')
    ax4.set_xticklabels(risk_metrics.index, rotation=45)
    ax4.legend()
    
    # 5. Distribution Analysis
    ax5 = fig.add_subplot(gs[2, 0])
    for asset_class in stats['Asset Class'].unique():
        mask = stats['Asset Class'] == asset_class
        returns[returns.columns[mask]].mean().hist(
            alpha=0.5,
            label=asset_class,
            bins=20,
            ax=ax5
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
        class_return.plot(label=asset_class, ax=ax6)
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
    """Create comprehensive visualization of efficient frontier results"""
    
    # Set up the plot grid
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2)
    
    # 1. Efficient Frontier Plot
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(
        frontier_results['risks'],
        frontier_results['returns'],
        c=frontier_results['sharpe_ratios'],
        cmap='viridis',
        s=100
    )
    plt.colorbar(sc, ax=ax1, label='Robust Sharpe Ratio')
    ax1.set_xlabel('Risk (Volatility)')
    ax1.set_ylabel('Expected Return')
    ax1.set_title(f'Robust Efficient Frontier (ε={epsilon:.2f})')
    ax1.grid(True, alpha=0.3)
    
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
    
    # 3. Risk Decomposition
    ax3 = fig.add_subplot(gs[1, 0])
    optimal_idx = np.argmax(frontier_results['sharpe_ratios'])
    optimal_weights = frontier_results['weights'][optimal_idx]
    
    # Calculate risk contribution for optimal portfolio
    cov = returns.cov().values
    total_risk = np.sqrt(optimal_weights @ cov @ optimal_weights)
    marginal_risk = (cov @ optimal_weights) / total_risk
    risk_contrib = optimal_weights * marginal_risk
    
    # Group by asset class
    class_risk = {}
    for i, col in enumerate(returns.columns):
        asset_class = asset_mapping[col]['class']
        class_risk[asset_class] = class_risk.get(asset_class, 0) + risk_contrib[i]
    
    pd.Series(class_risk).plot(kind='pie', autopct='%1.1f%%', ax=ax3)
    ax3.set_title('Risk Contribution (Optimal Portfolio)')
    
    # 4. Robust Metrics Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    metrics_df = pd.DataFrame({
        'Return': frontier_results['returns'],
        'Risk': frontier_results['risks'],
        'Robust Sharpe': frontier_results['sharpe_ratios'],
    })
    
    ax4.plot(metrics_df.index, metrics_df['Robust Sharpe'], 'b-', label='Robust Sharpe')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(metrics_df.index, metrics_df['Return'], 'r--', label='Return')
    ax4_twin.plot(metrics_df.index, metrics_df['Risk'], 'g--', label='Risk')
    
    ax4.set_xlabel('Portfolio Number')
    ax4.set_ylabel('Robust Sharpe Ratio')
    ax4_twin.set_ylabel('Return/Risk')
    ax4.set_title('Portfolio Metrics Evolution')
    
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 5. Optimal Portfolio Analysis
    ax5 = fig.add_subplot(gs[2, 0])
    optimal_alloc = {}
    for i, col in enumerate(returns.columns):
        asset_class = asset_mapping[col]['class']
        optimal_alloc[asset_class] = optimal_alloc.get(asset_class, 0) + optimal_weights[i]
    
    pd.Series(optimal_alloc).plot(kind='bar', ax=ax5)
    ax5.set_title('Optimal Portfolio Allocation')
    ax5.set_ylabel('Allocation')
    plt.xticks(rotation=45)
    
    # 6. Efficiency Analysis
    ax6 = fig.add_subplot(gs[2, 1])
    efficiency = frontier_results['returns'] / frontier_results['risks']
    ax6.plot(frontier_results['risks'], efficiency, 'b-')
    ax6.scatter(frontier_results['risks'], efficiency, c=frontier_results['sharpe_ratios'],
                cmap='viridis', s=100)
    ax6.set_xlabel('Risk (Volatility)')
    ax6.set_ylabel('Return/Risk Ratio')
    ax6.set_title('Portfolio Efficiency Analysis')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis continued
    def print_detailed_analysis(frontier_results: Dict[str, np.ndarray],
                              returns: pd.DataFrame,
                              asset_mapping: Dict):
        """Print comprehensive analysis of frontier results"""
        print("\nDetailed Portfolio Analysis")
        print("=" * 80)
        
        # Find optimal portfolio
        optimal_idx = np.argmax(frontier_results['sharpe_ratios'])
        optimal_weights = frontier_results['weights'][optimal_idx]
        
        print("\n1. Optimal Portfolio Characteristics:")
        print("-" * 40)
        print(f"Expected Return: {frontier_results['returns'][optimal_idx]:.4%}")
        print(f"Risk: {frontier_results['risks'][optimal_idx]:.4%}")
        print(f"Robust Sharpe Ratio: {frontier_results['sharpe_ratios'][optimal_idx]:.4f}")
        
        print("\n2. Asset Class Allocation:")
        print("-" * 40)
        class_alloc = {}
        for i, col in enumerate(returns.columns):
            asset_class = asset_mapping[col]['class']
            class_alloc[asset_class] = class_alloc.get(asset_class, 0) + optimal_weights[i]
        
        for asset_class, alloc in sorted(class_alloc.items()):
            print(f"{asset_class:20s}: {alloc:8.2%}")
        
        print("\n3. Individual Asset Weights:")
        print("-" * 40)
        for i, col in enumerate(returns.columns):
            if optimal_weights[i] > 0.01:  # Show only significant positions
                print(f"{col:8s} ({asset_mapping[col]['class']:20s}): {optimal_weights[i]:8.2%}")
        
        print("\n4. Risk Decomposition:")
        print("-" * 40)
        cov = returns.cov().values
        total_risk = np.sqrt(optimal_weights @ cov @ optimal_weights)
        marginal_risk = (cov @ optimal_weights) / total_risk
        risk_contrib = optimal_weights * marginal_risk
        
        class_risk = {}
        for i, col in enumerate(returns.columns):
            asset_class = asset_mapping[col]['class']
            class_risk[asset_class] = class_risk.get(asset_class, 0) + risk_contrib[i]
        
        for asset_class, risk in sorted(class_risk.items()):
            print(f"{asset_class:20s}: {risk:8.2%}")
        
        print("\n5. Frontier Characteristics:")
        print("-" * 40)
        print(f"Number of portfolios: {len(frontier_results['risks'])}")
        print(f"Risk range: {frontier_results['risks'].min():.4%} to {frontier_results['risks'].max():.4%}")
        print(f"Return range: {frontier_results['returns'].min():.4%} to {frontier_results['returns'].max():.4%}")
        
        # Calculate concentration metrics
        herfindahl = np.sum(optimal_weights ** 2)
        effective_n = 1 / herfindahl
        
        print("\n6. Portfolio Concentration Metrics:")
        print("-" * 40)
        print(f"Herfindahl Index: {herfindahl:.4f}")
        print(f"Effective N: {effective_n:.2f}")
        
        return {
            'optimal_weights': optimal_weights,
            'class_allocation': class_alloc,
            'risk_contribution': class_risk,
            'concentration': {
                'herfindahl': herfindahl,
                'effective_n': effective_n
            }
        }
    
    # Run the detailed analysis
    analysis_results = print_detailed_analysis(frontier_results, returns, asset_mapping)
    
    return analysis_results

def plot_risk_contribution_analysis(analysis_results: Dict,
                                  returns: pd.DataFrame,
                                  asset_mapping: Dict):
    """Create detailed risk contribution analysis plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Risk Contribution vs Allocation
    risk_contrib = pd.Series(analysis_results['risk_contribution'])
    allocation = pd.Series(analysis_results['class_allocation'])
    
    x = allocation.values
    y = risk_contrib.values
    ax1.scatter(x, y)
    ax1.plot([0, max(x)], [0, max(x)], 'r--', alpha=0.5)  # diagonal line
    
    for i, txt in enumerate(risk_contrib.index):
        ax1.annotate(txt, (x[i], y[i]))
    
    ax1.set_xlabel('Allocation')
    ax1.set_ylabel('Risk Contribution')
    ax1.set_title('Risk Contribution vs Allocation')
    ax1.grid(True, alpha=0.3)
    
    # 2. Correlation Structure (Fixed version)
    # Create asset class returns by first grouping columns by asset class
    asset_class_returns = pd.DataFrame()
    for asset_class in set(asset_info['class'] for asset_info in asset_mapping.values()):
        class_columns = [col for col in returns.columns 
                        if asset_mapping[col]['class'] == asset_class]
        asset_class_returns[asset_class] = returns[class_columns].mean(axis=1)
    
    # Calculate correlation matrix for asset classes
    corr = asset_class_returns.corr()
    sns.heatmap(corr, ax=ax2, cmap='RdYlBu_r', center=0, annot=True, fmt='.2f')
    ax2.set_title('Asset Class Correlation Structure')
    
    # 3. Risk Decomposition
    pd.Series(analysis_results['risk_contribution']).plot(
        kind='pie',
        autopct='%1.1f%%',
        ax=ax3
    )
    ax3.set_title('Risk Contribution Breakdown')
    
    # 4. Diversification Metrics
    metrics = pd.Series({
        'Herfindahl': analysis_results['concentration']['herfindahl'],
        'Effective N': analysis_results['concentration']['effective_n'],
        'Num Classes': len(analysis_results['class_allocation'])
    })
    
    metrics.plot(kind='bar', ax=ax4)
    ax4.set_title('Diversification Metrics')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calculate_frontier_risk_contributions(frontier_results: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Calculate and analyze risk contributions across the frontier"""
    n_points, n_assets = frontier_results['weights'].shape
    contributions = frontier_results['risk_contributions']
    
    return pd.DataFrame(
        contributions,
        columns=[f'Asset_{i}' for i in range(n_assets)],
        index=[f'Point_{i}' for i in range(n_points)]
    )

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
    
    # Plot risk contribution analysis
    print("\nGenerating risk analysis visualizations...")
    plot_risk_contribution_analysis(analysis_results, data['returns'], data['asset_mapping'])
    
    return {
        'universe_stats': universe_stats,
        'frontier_results': frontier_results,
        'analysis_results': analysis_results
    }

def run_efficient_frontier_analysis(returns: pd.DataFrame, asset_mapping: Dict):
    """Run efficient frontier analysis with robust optimization"""
    print("\nInitializing Robust Efficient Frontier Analysis...")
    
    # Define asset-specific risk aversion parameters
    alpha_by_class = {
        'Developed Equities': 1.2,    # Moderate risk aversion
        'Emerging Markets': 1.8,      # Higher risk aversion
        'Fixed Income': 0.8,          # Lower risk aversion
        'Alternative': 1.5            # Higher risk aversion
    }
    
    # Create alpha vector
    alphas = np.array([
        alpha_by_class[asset_mapping[col]['class']] 
        for col in returns.columns
    ])
    
    # Print configuration
    print("\nRisk Aversion Parameters:")
    for col in returns.columns:
        asset_class = asset_mapping[col]['class']
        print(f"{col} ({asset_class}): α={alpha_by_class[asset_class]:.2f}")
    
    try:
        # Initialize calculator
        calculator = RobustEfficientFrontier(
            returns=returns,
            epsilon=0.1,              # Uncertainty parameter
            alpha=alphas,             # Asset-specific risk aversion
            half_life=24,             # 2-year half-life for exponential weighting
            risk_free_rate=0.02,      # 2% risk-free rate
            transaction_cost=0.002     # 20bps transaction cost
        )
        
        # Define constraints
        constraints = OptimizationConstraints(
            long_only=True,
            box_constraints={
                i: (0.0, 0.25) for i in range(len(returns.columns))  # Max 25% per asset
            },
            group_constraints={
                'Developed Equities': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Developed Equities'],
                    bounds=(0.2, 0.5)  # 20-50% in developed markets
                ),
                'Fixed Income': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Fixed Income'],
                    bounds=(0.15, 0.4)  # 15-40% in fixed income
                ),
                'Emerging Markets': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Emerging Markets'],
                    bounds=(0.05, 0.25)  # 5-25% in emerging markets
                ),
                'Alternative': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Alternative'],
                    bounds=(0.05, 0.2)  # 5-20% in alternatives
                )
            }
        )
        
        print("\nComputing efficient frontier...")
        results = calculator.compute_efficient_frontier(
            n_points=10,              # Number of points on frontier
            epsilon_range=(0.05, 0.15),  # Range for uncertainty parameter
            alpha_scale_range=(0.8, 1.2),  # Range for scaling risk aversion
            constraints=constraints
        )
        
        # Print initial results
        print("\nFrontier Computation Complete:")
        print(f"Number of portfolios: {len(results['risks'])}")
        print(f"Risk range: {results['risks'].min():.4%} to {results['risks'].max():.4%}")
        print(f"Return range: {results['returns'].min():.4%} to {results['returns'].max():.4%}")
        
        return results
        
    except Exception as e:
        print(f"\nError in frontier computation: {str(e)}")
        raise

if __name__ == "__main__":
    results = run_complete_analysis()