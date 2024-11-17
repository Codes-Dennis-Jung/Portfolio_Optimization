import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from PortOpt import *
from typing import Dict
import os

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
    """Create visualization of efficient frontier results with tracking error analysis"""
    
    # Create output directory if it doesn't exist
    output_dir = 'Output_Efficient_Frontier'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated output directory: {output_dir}")
    
    # Set up the plot grid
    plt.style.use('seaborn')
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
    
    # Format axes as percentages
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
    
    # Format x-axis as percentage
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
    
    # Format axes as percentages
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'efficient_frontier_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.show()
    
    # Calculate and save key statistics
    optimal_idx = np.argmax(frontier_results['sharpe_ratios'])
    optimal_weights = frontier_results['weights'][optimal_idx]
    
    # Save optimal portfolio characteristics
    optimal_stats = {
        'Expected Return': frontier_results['returns'][optimal_idx],
        'Risk': frontier_results['risks'][optimal_idx],
        'Sharpe Ratio': frontier_results['sharpe_ratios'][optimal_idx],
        'Tracking Error': frontier_results['tracking_errors'][optimal_idx]
    }
    
    # Print and save optimal portfolio characteristics
    print("\nOptimal Portfolio Characteristics:")
    optimal_stats_df = pd.DataFrame.from_dict(optimal_stats, orient='index', columns=['Value'])
    optimal_stats_df.to_excel(os.path.join(output_dir, 'optimal_portfolio_stats.xlsx'))
    print(f"Expected Return: {optimal_stats['Expected Return']:.2%}")
    print(f"Risk: {optimal_stats['Risk']:.2%}")
    print(f"Sharpe Ratio: {optimal_stats['Sharpe Ratio']:.2f}")
    print(f"Tracking Error: {optimal_stats['Tracking Error']:.2%}")
    
    # Calculate and save asset class allocations
    class_alloc = {}
    for i, col in enumerate(returns.columns):
        asset_class = asset_mapping[col]['class']
        class_alloc[asset_class] = class_alloc.get(asset_class, 0) + optimal_weights[i]
    
    # Save optimal asset class allocations
    print("\nOptimal Asset Class Allocation:")
    class_alloc_df = pd.DataFrame.from_dict(class_alloc, orient='index', columns=['Allocation'])
    class_alloc_df.to_excel(os.path.join(output_dir, 'optimal_class_allocation.xlsx'))
    for asset_class, alloc in sorted(class_alloc.items()):
        print(f"{asset_class:20s}: {alloc:8.2%}")
    
    # Save detailed allocations evolution
    allocations_df = pd.DataFrame(
        allocations,
        columns=asset_classes,
        index=[f'Portfolio_{i}' for i in range(len(frontier_results['risks']))]
    )
    allocations_df.to_excel(os.path.join(output_dir, 'allocation_evolution.xlsx'))
    
    # Save frontier points with tracking errors
    frontier_df = pd.DataFrame({
        'Risk': frontier_results['risks'],
        'Return': frontier_results['returns'],
        'Sharpe_Ratio': frontier_results['sharpe_ratios'],
        'Tracking_Error': frontier_results['tracking_errors']
    })
    frontier_df.to_excel(os.path.join(output_dir, 'frontier_points.xlsx'))
    
    # Save individual asset weights
    weights_df = pd.DataFrame(
        frontier_results['weights'],
        columns=returns.columns,
        index=[f'Portfolio_{i}' for i in range(len(frontier_results['risks']))]
    )
    weights_df.to_excel(os.path.join(output_dir, 'portfolio_weights.xlsx'))
    
    print(f"\nAll results saved in: {os.path.abspath(output_dir)}")
    
    # Save summary statistics for the efficient frontier
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
                i: (0.0, 0.25) for i in range(len(returns.columns))  # Max 25% per asset
            },
            group_constraints={
                'Developed Equities': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Developed Equities'],
                    bounds=(0.01, 0.5)  # 1-50% in developed markets
                ),
                'Fixed Income': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Fixed Income'],
                    bounds=(0.01, 0.4)  # 1-40% in fixed income
                ),
                'Emerging Markets': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Emerging Markets'],
                    bounds=(0.01, 0.25)  # 1-25% in emerging markets
                ),
                'Alternative': GroupConstraint(
                    assets=[i for i, col in enumerate(returns.columns)
                           if asset_mapping[col]['class'] == 'Alternative'],
                    bounds=(0.01, 0.2)  # 1-20% in alternatives
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
            n_points=20,             
            epsilon_range=(0.05, 0.5),
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