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
        lookback_window=36,           # 2 years
        rebalance_frequency=3,        # Quarterly
        epsilon=0.1,                  # Uncertainty parameter
        transaction_cost=0.001,       # 10bps per trade
        risk_free_rate=universe_data['risk_free_rate'],
        use_cross_validation=False,   # Disable CV for testing
        estimation_method='standard',
        risk_estimator='empirical'
    )
    
    # Define portfolio constraints
    constraints = OptimizationConstraints(
        long_only=True,
        box_constraints={i: (0.0, 0.5) for i in range(len(returns.columns))},  # Max 30% per asset
        group_constraints=group_constraints,
        max_turnover=0.5  # 50% max turnover per rebalance
    )
    
    # Define strategies to test (start with just minimum variance)
    strategies = [
        (ObjectiveFunction.MINIMUM_VARIANCE, "Minimum Variance", {}),
        (ObjectiveFunction.GARLAPPI_ROBUST, "Garlappi Robust", {}),
        (ObjectiveFunction.MEAN_VARIANCE, "Mean Variance", {}),
        (ObjectiveFunction.MAXIMUM_DIVERSIFICATION, "Max Diversifiction Variance", {})
    ]
    
    print("\nStep 5: Running backtests...")
    results = {}
    successful_strategies = []
    
    for objective, name, params in strategies:
        print(f"\nTesting {name} strategy...")
        try:
            # Initialize with equal weights
            initial_weights = pd.Series(
                np.ones(len(returns.columns)) / len(returns.columns),
                index=returns.columns
            )
            
            # Run backtest
            strategy_result = backtester.run_backtest(
                objective=objective,
                constraints=constraints,
                current_weights=initial_weights,
                **params
            )
            
            #filename_ = os.getcwd() + str(name)
            #backtester.save_backtest_results(strategy_result,filename_)
            
            # Print strategy summary
            metrics = strategy_result['backtest_metrics']
            print(f"\n{name} Performance Summary:")
            print(f"Annualized Return: {float(metrics['Annualized Return']):.2%}")
            print(f"Volatility: {float(metrics['Volatility']):.2%}")
            print(f"Sharpe Ratio: {float(metrics['Sharpe Ratio']):.2f}")
            print(f"Maximum Drawdown: {float(metrics['Maximum Drawdown']):.2%}")
            print(f"Average Turnover: {float(metrics['Average Turnover']):.2%}")
            
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

def run_multi_asset_test():
    """Main function to run multi-asset backtest test"""
    try:
        print("Starting multi-asset backtest test...")
        test_results = test_multi_asset_backtest()
        
        if test_results is not None:
            print("\nTest completed successfully!")
            
            # Print detailed analysis
            print("\nStrategy Asset Class Exposures:")
            exposures = test_results['comparison'].filter(regex='Exposure$')
            print(exposures.round(3))
            
            # Plot additional analysis
            plot_additional_analysis(test_results)
            
            return test_results
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
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
    """
    summary = pd.DataFrame()
    
    for strategy_name, result in results.items():
        metrics = result['backtest_metrics']
        
        # Basic metrics (ensure all values are float)
        summary.loc[strategy_name, 'Total Return'] = float(metrics['Total Return'])
        summary.loc[strategy_name, 'Ann. Return'] = float(metrics['Annualized Return'])
        summary.loc[strategy_name, 'Ann. Volatility'] = float(metrics['Volatility'])
        summary.loc[strategy_name, 'Sharpe Ratio'] = float(metrics['Sharpe Ratio'])
        summary.loc[strategy_name, 'Max Drawdown'] = float(metrics['Maximum Drawdown'])
        summary.loc[strategy_name, 'Avg Turnover'] = float(metrics['Average Turnover'])
        
        # Calculate average asset class exposures
        weights_df = result['weights']
        avg_weights = weights_df.mean()
        
        for asset_class in set(info['class'] for info in asset_mapping.values()):
            class_weight = sum(
                avg_weights[symbol] 
                for symbol in avg_weights.index
                if asset_mapping[symbol]['class'] == asset_class
            )
            summary.loc[strategy_name, f'{asset_class} Exposure'] = float(class_weight)
    
    return summary

def plot_strategy_comparison(
    results: Dict[str, Dict],
    universe_data: Dict
):
    """Create comprehensive visualization of strategy comparison with proper date handling"""
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2)
    
    # Convert timestamps to datetime for plotting
    def prepare_dates(index):
        return pd.to_datetime(index).tz_localize(None)
    
    # 1. Cumulative Returns
    ax1 = fig.add_subplot(gs[0, :])
    for strategy_name, result in results.items():
        cum_returns = (1 + result['returns']).cumprod()
        ax1.plot(prepare_dates(cum_returns.index), 
                cum_returns.values, 
                label=strategy_name, 
                linewidth=2)
    ax1.set_title('Cumulative Strategy Returns')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 2. Drawdowns
    ax2 = fig.add_subplot(gs[1, 0])
    for strategy_name, result in results.items():
        cum_returns = (1 + result['returns']).cumprod()
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        ax2.plot(prepare_dates(drawdown.index), 
                drawdown.values, 
                label=strategy_name)
    ax2.set_title('Strategy Drawdowns')
    ax2.legend(loc='lower left')
    ax2.grid(True)
    
    # 3. Rolling Sharpe Ratios
    ax3 = fig.add_subplot(gs[1, 1])
    window = 24  # 2-year rolling window
    rf_rate = universe_data['risk_free_rate']
    for strategy_name, result in results.items():
        returns = result['returns']
        rolling_ret = returns.rolling(window).mean() * 12
        rolling_vol = returns.rolling(window).std() * np.sqrt(12)
        rolling_sharpe = (rolling_ret - rf_rate * 12) / rolling_vol
        ax3.plot(prepare_dates(rolling_sharpe.index), 
                rolling_sharpe.values, 
                label=strategy_name)
    ax3.set_title('Rolling 2-Year Sharpe Ratio')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    # 4. Asset Class Exposures
    ax4 = fig.add_subplot(gs[2, 0])
    asset_mapping = universe_data['asset_mapping']
    asset_classes = sorted(set(info['class'] for info in asset_mapping.values()))
    
    strategy_exposures = []
    strategy_names = []
    
    for strategy_name, result in results.items():
        weights_df = result['weights']
        avg_weights = weights_df.mean()
        exposures = []
        
        for asset_class in asset_classes:
            class_weight = sum(
                avg_weights[symbol] for symbol in avg_weights.index
                if asset_mapping[symbol]['class'] == asset_class
            )
            exposures.append(class_weight)
            
        strategy_exposures.append(exposures)
        strategy_names.append(strategy_name)
    
    exposures_df = pd.DataFrame(
        strategy_exposures,
        columns=asset_classes,
        index=strategy_names
    )
    
    exposures_df.plot(kind='bar', stacked=True, ax=ax4)
    ax4.set_title('Average Asset Class Exposures')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    
    # 5. Risk-Return Scatter
    ax5 = fig.add_subplot(gs[2, 1])
    ann_returns = []
    ann_vols = []
    
    for strategy_name, result in results.items():
        returns = result['returns']
        ann_ret = float(result['backtest_metrics']['Annualized Return'])
        ann_vol = float(result['backtest_metrics']['Volatility'])
        ann_returns.append(ann_ret)
        ann_vols.append(ann_vol)
        
        ax5.scatter(ann_vol, ann_ret, s=100)
        ax5.annotate(
            strategy_name, 
            (ann_vol, ann_ret),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    ax5.set_xlabel('Annualized Volatility')
    ax5.set_ylabel('Annualized Return')
    ax5.set_title('Risk-Return Comparison')
    ax5.grid(True)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def plot_additional_analysis(test_results: Dict):
    """Plot additional analysis charts"""
    if test_results is None:
        return
        
    # Plot cumulative returns comparison
    plt.figure(figsize=(12, 6))
    for name, result in test_results['results'].items():
        cum_returns = (1 + result['returns']).cumprod()
        # Convert timestamps for plotting
        dates = pd.to_datetime(cum_returns.index).tz_localize(None)
        plt.plot(dates, cum_returns.values, label=name)
    
    plt.title('Cumulative Strategy Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot rolling correlation heatmap if multiple strategies
    if len(test_results['results']) > 1:
        returns_df = pd.DataFrame({
            name: result['returns']
            for name, result in test_results['results'].items()
        })
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            returns_df.corr(),
            annot=True,
            cmap='RdYlBu_r',
            center=0,
            vmin=-1,
            vmax=1
        )
        plt.title('Strategy Return Correlations')
        plt.show()

if __name__ == "__main__":
    try:
        print("Starting multi-asset backtest test...")
        test_results = test_multi_asset_backtest()
        
        if test_results is not None:
            print("\nTest completed successfully!")
            
            # Additional analysis
            print("\nAsset Class Exposures:")
            exposures = test_results['comparison'].filter(regex='Exposure$')
            print(exposures.round(3))
            
            # Plot additional analysis
            plot_additional_analysis(test_results)
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
