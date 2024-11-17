import pandas as pd
import seaborn as sns
import os
import matplotlib as plt
from PortOpt import *

################################################################################################################
#### Testing Portfolio Backtest
################################################################################################################

# Loading test data 
def get_multi_asset_test_data():
    """Get multi-asset universe data for testing"""
    import yfinance as yf
    from datetime import datetime, timedelta
    
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
            'Alternatives': ['ABRYX', 'QAI']   # Absolute Return, Multi-Strategy
        }
    }
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15*365)  # 6 years of data
    
    # Download data
    prices = pd.DataFrame()
    asset_mapping = {}  # To store asset class mapping
    failed_downloads = []
    
    print("Downloading multi-asset data...")
    for asset_class, sub_classes in assets.items():
        for sub_class, symbols in sub_classes.items():
            for symbol in symbols:
                try:
                    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if len(data) > 0:
                        prices[symbol] = data['Adj Close']
                        asset_mapping[symbol] = {'class': asset_class, 'sub_class': sub_class}
                    else:
                        failed_downloads.append(symbol)
                except Exception as e:
                    print(f"Failed to download {symbol}: {e}")
                    failed_downloads.append(symbol)
    
    if failed_downloads:
        print(f"Warning: Failed to download data for: {', '.join(failed_downloads)}")
    
    # Resample to monthly frequency
    prices = prices.resample('M').last()
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Handle missing data and outliers
    returns = returns.fillna(method='ffill').fillna(method='bfill')
    
    # Remove extreme outliers using Winsorization
    for col in returns.columns:
        percentiles = returns[col].quantile([0.01, 0.99])
        returns[col] = returns[col].clip(percentiles[0.01], percentiles[0.99])
    
    # Get risk-free rate data
    try:
        tickers = yf.Tickers('^IRX')  # 13-week Treasury Bill rate
        rf_data = tickers.history(start=start_date, end=end_date)['Close'] / 100 / 12  # Monthly rate
        rf_rate = rf_data.mean()
    except:
        print("Warning: Using default risk-free rate")
        rf_rate = 0.02 / 12  # Default monthly risk-free rate
    
    # Calculate basic statistics
    print("\nAsset Universe Summary:")
    print(f"Number of assets: {len(returns.columns)}")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    print(f"Number of periods: {len(returns)}")
    
    # Print asset class composition
    print("\nAsset Class Composition:")
    for asset_class in assets.keys():
        count = sum(1 for v in asset_mapping.values() if v['class'] == asset_class)
        print(f"{asset_class}: {count} assets")
    
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    # Print correlation summary
    print("\nCorrelation Summary:")
    print(f"Average correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
    
    return {
        'returns': returns,
        'asset_mapping': asset_mapping,
        'correlation': corr_matrix,
        'risk_free_rate': rf_rate,
        'prices': prices
    }

def analyze_universe(data: Dict):
    """Analyze the multi-asset universe"""
    returns = data['returns']
    asset_mapping = data['asset_mapping']
    
    # Calculate asset class level statistics
    stats = pd.DataFrame(columns=['Annual Return', 'Annual Vol', 'Sharpe Ratio', 'Max Drawdown'])
    
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
        
        stats.loc[symbol] = [annual_return, annual_vol, sharpe, max_drawdown]
    
    # Add asset class information
    stats['Asset Class'] = [asset_mapping[symbol]['class'] for symbol in stats.index]
    stats['Sub Class'] = [asset_mapping[symbol]['sub_class'] for symbol in stats.index]
    
    # Print summary statistics by asset class
    print("\nAsset Class Summary Statistics:")
    print(stats.groupby('Asset Class').mean().round(3))
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(2, 2)
    
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
    ax1.grid(True)
    
    # 2. Correlation Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(data['correlation'], cmap='RdYlBu_r', center=0, ax=ax2)
    ax2.set_title('Correlation Matrix')
    
    # 3. Drawdown Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    (-stats.groupby('Asset Class')['Max Drawdown'].mean()).plot(
        kind='bar',
        ax=ax3,
        color='darkred'
    )
    ax3.set_title('Average Maximum Drawdown by Asset Class')
    ax3.set_ylabel('Maximum Drawdown')
    plt.xticks(rotation=45)
    
    # 4. Sharpe Ratio Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    stats.groupby('Asset Class')['Sharpe Ratio'].mean().plot(
        kind='bar',
        ax=ax4,
        color='darkgreen'
    )
    ax4.set_title('Average Sharpe Ratio by Asset Class')
    ax4.set_ylabel('Sharpe Ratio')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return stats

# Example usage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from PortOpt import *

def test_multi_asset_backtest():
    """Test multi-asset portfolio optimization backtest"""
    # Step 1: Load and analyze universe data
    print("Step 1: Loading and analyzing universe data...")
    universe_data = get_multi_asset_test_data()
    returns = universe_data['returns']
    asset_mapping = universe_data['asset_mapping']
    
    # Analyze universe characteristics
    print("\nStep 2: Analyzing universe characteristics...")
    universe_stats = analyze_universe(universe_data)
    
    print("\nStep 3: Setting up optimization constraints...")
    # Create group constraints based on asset classes
    group_constraints = {}
    asset_classes = set(info['class'] for info in asset_mapping.values())
    
    # Create mapping of column names to indices
    col_to_idx = {col: i for i, col in enumerate(returns.columns)}
    
    # Set up asset class constraints
    for asset_class in asset_classes:
        assets_in_class = [
            col_to_idx[symbol] for symbol in returns.columns
            if asset_mapping[symbol]['class'] == asset_class
        ]
        
        if assets_in_class:
            group_constraints[asset_class] = GroupConstraint(
                assets=assets_in_class,
                bounds=(0.05, 0.4)  # 5% min, 40% max per asset class
            )
    
    # Initialize backtester
    print("\nStep 4: Initializing backtester...")
    backtester = RobustBacktestOptimizer(
        returns=returns,
        lookback_window=36,           # 3 years
        rebalance_frequency=3,        # Quarterly
        estimation_method='robust',    # Use robust estimation
        transaction_cost=0.001,       # 10bps per trade
        risk_free_rate=universe_data['risk_free_rate'],
        epsilon=0.1                   # Uncertainty parameter
    )
    
    # Define portfolio constraints
    constraints = OptimizationConstraints(
        long_only=True,
        box_constraints={i: (0.0, 0.5) for i in range(len(returns.columns))},  # Max 50% per asset
        group_constraints=group_constraints
    )
    
    # Define strategies to test
    strategies = [
        (ObjectiveFunction.MINIMUM_VARIANCE, "Minimum Variance", {}),
        (ObjectiveFunction.GARLAPPI_ROBUST, "Garlappi Robust", {'epsilon': 0.1}),
        (ObjectiveFunction.MEAN_VARIANCE, "Mean Variance", {}),
        (ObjectiveFunction.MAXIMUM_DIVERSIFICATION, "Maximum Diversification", {})
    ]
    
    print("\nStep 5: Running backtests...")
    results = {}
    successful_strategies = []
    
    for objective, name, params in strategies:
        print(f"\nTesting {name} strategy...")
        try:
            # Initialize with equal weights
            initial_weights = np.ones(len(returns.columns)) / len(returns.columns)
            
            # Run backtest
            strategy_result = backtester.run_backtest(
                objective=objective,
                constraints=constraints,
                initial_weights=initial_weights,
                **params
            )
            
            # Save results
            filename = f"{name.replace(' ', '_')}_results.xlsx"
            backtester.save_backtest_results(strategy_result, filename)
            print(f"Results saved to {filename}")
            
            # Print strategy summary with corrected metrics access
            metrics_df = strategy_result['backtest_metrics']
            print(f"\n{name} Performance Summary:")
            # Convert Series to float before formatting
            print(f"Total Return: {float(metrics_df.loc['Total Return', 'value']):.2%}")
            print(f"Annualized Return: {float(metrics_df.loc['Annualized Return', 'value']):.2%}")
            print(f"Volatility: {float(metrics_df.loc['Volatility', 'value']):.2%}")
            print(f"Sharpe Ratio: {float(metrics_df.loc['Sharpe Ratio', 'value']):.2f}")
            print(f"Maximum Drawdown: {float(metrics_df.loc['Maximum Drawdown', 'value']):.2%}")
            print(f"Average Turnover: {float(metrics_df.loc['Average Turnover', 'value']):.2%}")
            print(f"Total Costs: {float(metrics_df.loc['Total Costs', 'value']):.2%}")
            
            # Store results
            results[name] = strategy_result
            successful_strategies.append(name)
            
            # Print asset class exposures
            weights = strategy_result['weights'].mean()
            print("\nAverage Asset Class Exposures:")
            for asset_class in asset_classes:
                exposure = sum(
                    weights[symbol] for symbol in weights.index
                    if asset_mapping[symbol]['class'] == asset_class
                )
                print(f"{asset_class}: {exposure:.1%}")
            
        except Exception as e:
            print(f"Error testing {name} strategy: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    if successful_strategies:
        print("\nStep 6: Analyzing results...")
        try:
            # Compare strategies
            comparison = analyze_strategy_results(
                {k: results[k] for k in successful_strategies},
                asset_mapping
            )
            
            print("\nStrategy Comparison Summary:")
            print(comparison.round(3))
            
            # Plot results
            print("\nStep 7: Generating visualizations...")
            plot_strategy_comparison(
                {k: results[k] for k in successful_strategies},
                universe_data
            )
            
            return {
                'results': results,
                'comparison': comparison,
                'universe_data': universe_data,
                'universe_stats': universe_stats
            }
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
    else:
        print("No strategies completed successfully.")
        return None

def analyze_strategy_results(
    results: Dict[str, Dict],
    asset_mapping: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Analyze and compare strategy performance
    
    Args:
        results: Dictionary of strategy results
        asset_mapping: Asset class mapping information
        
    Returns:
        DataFrame with strategy performance metrics
    """
    summary = pd.DataFrame()
    
    for strategy_name, result in results.items():
        # Get metrics from backtest_metrics DataFrame
        metrics_df = result['backtest_metrics']
        
        # Basic metrics
        summary.loc[strategy_name, 'Total Return'] = float(metrics_df.loc['Total Return', 'value'])
        summary.loc[strategy_name, 'Annualized Return'] = float(metrics_df.loc['Annualized Return', 'value'])
        summary.loc[strategy_name, 'Volatility'] = float(metrics_df.loc['Volatility', 'value'])
        summary.loc[strategy_name, 'Sharpe Ratio'] = float(metrics_df.loc['Sharpe Ratio', 'value'])
        summary.loc[strategy_name, 'Maximum Drawdown'] = float(metrics_df.loc['Maximum Drawdown', 'value'])
        summary.loc[strategy_name, 'Average Turnover'] = float(metrics_df.loc['Average Turnover', 'value'])
        summary.loc[strategy_name, 'Total Costs'] = float(metrics_df.loc['Total Costs', 'value'])
        
        # Calculate asset class exposures
        weights_df = result['weights']
        avg_weights = weights_df.mean()
        
        for asset_class in set(info['class'] for info in asset_mapping.values()):
            class_weight = sum(
                avg_weights[symbol] 
                for symbol in avg_weights.index
                if asset_mapping[symbol]['class'] == asset_class
            )
            summary.loc[strategy_name, f'{asset_class} Exposure'] = float(class_weight)
        
        # Additional risk metrics
        returns_series = result['returns']['returns']
        
        # Calculate rolling metrics (36-month window)
        window = 36
        rolling_ret = returns_series.rolling(window=window).mean() * 12
        rolling_vol = returns_series.rolling(window=window).std() * np.sqrt(12)
        
        # Average metrics
        summary.loc[strategy_name, 'Avg Rolling Return'] = rolling_ret.mean()
        summary.loc[strategy_name, 'Avg Rolling Vol'] = rolling_vol.mean()
        
        # Risk metrics
        returns_array = returns_series.values
        summary.loc[strategy_name, 'Skewness'] = scipy.stats.skew(returns_array)
        summary.loc[strategy_name, 'Kurtosis'] = scipy.stats.kurtosis(returns_array)
        
        # Downside risk
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) > 0:
            summary.loc[strategy_name, 'Downside Vol'] = np.std(negative_returns) * np.sqrt(12)
            summary.loc[strategy_name, 'Max Monthly Loss'] = np.min(returns_array)
        else:
            summary.loc[strategy_name, 'Downside Vol'] = 0
            summary.loc[strategy_name, 'Max Monthly Loss'] = 0
            
        # Calculate drawdown statistics
        cum_returns = (1 + returns_series).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        
        summary.loc[strategy_name, 'Avg Drawdown'] = drawdowns.mean()
        summary.loc[strategy_name, 'Drawdown Duration'] = calculate_avg_drawdown_duration(drawdowns)
        
    return summary

def calculate_avg_drawdown_duration(drawdowns: pd.Series) -> float:
    """Calculate average drawdown duration in months"""
    if drawdowns.empty:
        return 0
        
    # Identify drawdown periods
    is_drawdown = drawdowns < 0
    
    if not is_drawdown.any():
        return 0
        
    # Find start and end of drawdowns
    starts = is_drawdown[is_drawdown != is_drawdown.shift(1)].index[::2]
    ends = is_drawdown[is_drawdown != is_drawdown.shift(1)].index[1::2]
    
    if len(starts) == 0 or len(ends) == 0:
        return 0
        
    # Calculate durations
    durations = [(end - start).days / 30.44 for start, end in zip(starts, ends)]  # Convert to months
    
    return np.mean(durations) if durations else 0

def plot_strategy_comparison(
    results: Dict[str, Dict],
    universe_data: Dict,
    figsize: Tuple[int, int] = (20, 15)
):
    """Create comprehensive visualization of strategy comparison"""
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(3, 2)
    
    # 1. Cumulative Returns
    ax1 = fig.add_subplot(gs[0, :])
    for strategy_name, result in results.items():
        cumulative = (1 + result['returns']['returns']).cumprod()
        ax1.plot(cumulative.index, cumulative.values, label=strategy_name, linewidth=2)
    ax1.set_title('Cumulative Strategy Returns')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 2. Rolling Volatility
    ax2 = fig.add_subplot(gs[1, 0])
    window = 36  # 3-year rolling window
    for strategy_name, result in results.items():
        rolling_vol = result['returns']['returns'].rolling(window).std() * np.sqrt(12)
        ax2.plot(rolling_vol.index, rolling_vol.values, label=strategy_name)
    ax2.set_title('Rolling 3-Year Volatility')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # 3. Rolling Sharpe Ratio
    ax3 = fig.add_subplot(gs[1, 1])
    rf_rate = universe_data['risk_free_rate']
    for strategy_name, result in results.items():
        returns = result['returns']['returns']
        rolling_ret = returns.rolling(window).mean() * 12
        rolling_vol = returns.rolling(window).std() * np.sqrt(12)
        rolling_sharpe = (rolling_ret - rf_rate * 12) / rolling_vol
        ax3.plot(rolling_sharpe.index, rolling_sharpe.values, label=strategy_name)
    ax3.set_title('Rolling 3-Year Sharpe Ratio')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    # 4. Drawdowns
    ax4 = fig.add_subplot(gs[2, 0])
    for strategy_name, result in results.items():
        returns = result['returns']['returns']
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        ax4.plot(drawdowns.index, drawdowns.values, label=strategy_name)
    ax4.set_title('Strategy Drawdowns')
    ax4.legend(loc='lower left')
    ax4.grid(True)
    
    # 5. Asset Class Exposures
    ax5 = fig.add_subplot(gs[2, 1])
    exposures_data = []
    strategy_names = []
    asset_classes = sorted(set(info['class'] for info in universe_data['asset_mapping'].values()))
    
    for strategy_name, result in results.items():
        weights = result['weights'].mean()
        exposures = []
        for asset_class in asset_classes:
            exposure = sum(
                weights[symbol] for symbol in weights.index
                if universe_data['asset_mapping'][symbol]['class'] == asset_class
            )
            exposures.append(exposure)
        exposures_data.append(exposures)
        strategy_names.append(strategy_name)
    
    exposure_df = pd.DataFrame(
        exposures_data,
        columns=asset_classes,
        index=strategy_names
    )
    
    exposure_df.plot(kind='bar', stacked=True, ax=ax5)
    ax5.set_title('Average Asset Class Exposures')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_results = test_multi_asset_backtest()