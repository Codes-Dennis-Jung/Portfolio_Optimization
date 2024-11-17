
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats
from typing import Dict,Tuple
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
def plot_strategy_comparison(
    results: Dict[str, Dict],
    universe_data: Dict,
    figsize: Tuple[int, int] = (20, 10),
    output_dir: str = r'Output_Backtest'
):
    """Create visualization of cumulative returns and drawdowns and save to file"""
    import datetime
    import os
    from matplotlib.dates import YearLocator, DateFormatter
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Create figure for plots
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 1)
    
    # Helper function to filter valid dates
    def get_valid_dates(index):
        return pd.DatetimeIndex([x for x in index if isinstance(x, (pd.Timestamp, datetime.datetime))])
    
    # 1. Cumulative Returns
    ax1 = fig.add_subplot(gs[0])
    for strategy_name, result in results.items():
        try:
            returns = result['returns']['returns']
            valid_dates = get_valid_dates(returns.index)
            cumulative = (1 + returns.loc[valid_dates]).cumprod()
            ax1.plot(cumulative.index, cumulative.values, label=strategy_name, linewidth=2)
            
            # Save cumulative returns data to Excel
            excel_path = os.path.join(output_dir, f"{strategy_name.replace(' ', '_')}_cumulative_returns.xlsx")
            cumulative.to_excel(excel_path)
            print(f"Saved cumulative returns to: {excel_path}")
        except Exception as e:
            print(f"Error plotting cumulative returns for {strategy_name}: {str(e)}")
    ax1.set_title('Cumulative Strategy Returns')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 2. Drawdowns
    ax2 = fig.add_subplot(gs[1])
    for strategy_name, result in results.items():
        try:
            returns = result['returns']['returns']
            valid_dates = get_valid_dates(returns.index)
            returns_clean = returns.loc[valid_dates]
            
            cum_returns = (1 + returns_clean).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            
            ax2.plot(drawdowns.index, drawdowns.values, label=strategy_name)
            
            # Save drawdowns data to Excel
            excel_path = os.path.join(output_dir, f"{strategy_name.replace(' ', '_')}_drawdowns.xlsx")
            drawdowns.to_excel(excel_path)
            print(f"Saved drawdowns to: {excel_path}")
        except Exception as e:
            print(f"Error plotting drawdowns for {strategy_name}: {str(e)}")
    ax2.set_title('Strategy Drawdowns')
    ax2.legend(loc='lower left')
    ax2.grid(True)
    
    # Format date axes
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save plot to file
    plot_path = os.path.join(output_dir, 'strategy_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plots to: {plot_path}")
    
    plt.show()

def test_multi_asset_backtest():
    """Test multi-asset portfolio optimization backtest"""
    # Create output directory
    output_dir = r'Output_Backtest'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
    
    # Initialize base constraints
    constraints = OptimizationConstraints(
        long_only=True,
        box_constraints={i: (0.0, 0.3) for i in range(len(returns.columns))},
        group_constraints=group_constraints
    )
    
    # Define strategies
    strategies = [
        (ObjectiveFunction.MINIMUM_VARIANCE, "Minimum Variance", {}),
        (ObjectiveFunction.GARLAPPI_ROBUST, "Garlappi Robust", {
            'epsilon': 0.1,  # Uncertainty parameter
            'alpha': np.ones(len(returns.columns)),  # Asset-specific risk aversion
            'omega_method': 'bayes'  # Estimation method for Omega
        }),
        (ObjectiveFunction.MEAN_VARIANCE, "Mean Variance", {}),
        (ObjectiveFunction.MAXIMUM_DIVERSIFICATION, "Maximum Diversification", {}),
        (ObjectiveFunction.ROBUST_MEAN_VARIANCE, "Robust Mean-Variance",  {
            'epsilon': 0.1, 'kappa': 1.0}),
        (ObjectiveFunction.MAXIMUM_SHARPE, "Maximum Sharpe", {}),
        (ObjectiveFunction.RISK_PARITY, "Risk Parity", {}),
        (ObjectiveFunction.MEAN_CVAR, "Mean-CVaR", {}),
        (ObjectiveFunction.HIERARCHICAL_RISK_PARITY, "Hierarchical Risk Parity", {}),
        (ObjectiveFunction.EQUAL_RISK_CONTRIBUTION, "Equal Risk Contribution", {})
    ]
    
    print("\nStep 4: Running backtests...")
    results = {}
    successful_strategies = []
    
    for objective, name, params in strategies:
        print(f"\nTesting {name} strategy...")
        try:
            backtester = RobustBacktestOptimizer(
                returns=returns,
                lookback_window=36,
                rebalance_frequency=1,
                estimation_method='robust',
                transaction_cost=0.001,
                risk_free_rate=universe_data['risk_free_rate'],
                epsilon=params.get('epsilon', 0.1)
            )
            
            initial_weights = np.ones(len(returns.columns)) / len(returns.columns)
            
            try:
                strategy_result = backtester.run_backtest(
                    objective=objective,
                    constraints=constraints,
                    initial_weights=initial_weights,
                    **params
                )
                
                # Save results with proper path joining
                filename = os.path.join(output_dir, f"{name.replace(' ', '_')}_results.xlsx")
                backtester.save_backtest_results(strategy_result, filename)
                print(f"Results saved to: {filename}")
                
                # Print strategy summary
                metrics_df = strategy_result['backtest_metrics']
                print(f"\n{name} Performance Summary:")
                for metric in ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 
                             'Maximum Drawdown', 'Average Turnover', 'Total Costs']:
                    print(f"{metric}: {float(metrics_df.loc[metric, 'value']):.2%}")
                
                results[name] = strategy_result
                successful_strategies.append(name)
                
            except Exception as e:
                print(f"Error in backtest for {name}: {str(e)}")
                print(traceback.format_exc())
                
        except Exception as e:
            print(f"Error initializing {name} strategy: {str(e)}")
            print(traceback.format_exc())
    
    if successful_strategies:
        print("\nStep 5: Generating visualizations and saving results...")
        try:
            # Plot and save results
            plot_strategy_comparison(
                {k: results[k] for k in successful_strategies},
                universe_data,
                output_dir=output_dir
            )
            
            return {
                'results': results,
                'universe_data': universe_data,
                'universe_stats': universe_stats
            }
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            print(traceback.format_exc())
            return None
    else:
        print("No strategies completed successfully.")
        return None
    
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
    figsize: Tuple[int, int] = (20, 10),
    output_dir: str = r'Output_Backtest'
):
    """Create visualization of cumulative returns and drawdowns and save to file"""
    import datetime
    import os
    from matplotlib.dates import YearLocator, DateFormatter
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Create figure for plots
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 1)
    
    # Helper function to filter valid dates
    def get_valid_dates(index):
        return pd.DatetimeIndex([x for x in index if isinstance(x, (pd.Timestamp, datetime.datetime))])
    
    # 1. Cumulative Returns
    ax1 = fig.add_subplot(gs[0])
    for strategy_name, result in results.items():
        try:
            returns = result['returns']['returns']
            valid_dates = get_valid_dates(returns.index)
            cumulative = (1 + returns.loc[valid_dates]).cumprod()
            ax1.plot(cumulative.index, cumulative.values, label=strategy_name, linewidth=2)
            
            # Save cumulative returns data to Excel
            excel_path = os.path.join(output_dir, f"{strategy_name.replace(' ', '_')}_cumulative_returns.xlsx")
            cumulative.to_excel(excel_path)
            print(f"Saved cumulative returns to: {excel_path}")
        except Exception as e:
            print(f"Error plotting cumulative returns for {strategy_name}: {str(e)}")
    ax1.set_title('Cumulative Strategy Returns')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 2. Drawdowns
    ax2 = fig.add_subplot(gs[1])
    for strategy_name, result in results.items():
        try:
            returns = result['returns']['returns']
            valid_dates = get_valid_dates(returns.index)
            returns_clean = returns.loc[valid_dates]
            
            cum_returns = (1 + returns_clean).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            
            ax2.plot(drawdowns.index, drawdowns.values, label=strategy_name)
            
            # Save drawdowns data to Excel
            excel_path = os.path.join(output_dir, f"{strategy_name.replace(' ', '_')}_drawdowns.xlsx")
            drawdowns.to_excel(excel_path)
            print(f"Saved drawdowns to: {excel_path}")
        except Exception as e:
            print(f"Error plotting drawdowns for {strategy_name}: {str(e)}")
    ax2.set_title('Strategy Drawdowns')
    ax2.legend(loc='lower left')
    ax2.grid(True)
    
    # Format date axes
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save plot to file
    plot_path = os.path.join(output_dir, 'strategy_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plots to: {plot_path}")
    
    plt.show()

def analyze_strategy_results(
    results: Dict[str, Dict],
    asset_mapping: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Analyze and compare strategy performance with improved error handling
    """
    summary = pd.DataFrame()
    
    for strategy_name, result in results.items():
        try:
            # Get metrics from backtest_metrics DataFrame
            metrics_df = result['backtest_metrics']
            
            # Basic metrics with error handling
            for metric in ['Total Return', 'Annualized Return', 'Volatility', 
                        'Sharpe Ratio', 'Maximum Drawdown', 'Average Turnover', 
                        'Total Costs']:
                try:
                    summary.loc[strategy_name, metric] = float(metrics_df.loc[metric, 'value'])
                except Exception as e:
                    print(f"Error calculating {metric} for {strategy_name}: {str(e)}")
                    summary.loc[strategy_name, metric] = np.nan
            
            # Calculate asset class exposures
            try:
                weights_df = result['weights']
                avg_weights = weights_df.mean()
                
                for asset_class in set(info['class'] for info in asset_mapping.values()):
                    class_weight = sum(
                        avg_weights[symbol] 
                        for symbol in avg_weights.index
                        if asset_mapping[symbol]['class'] == asset_class
                    )
                    summary.loc[strategy_name, f'{asset_class} Exposure'] = float(class_weight)
            except Exception as e:
                print(f"Error calculating exposures for {strategy_name}: {str(e)}")
            
            # Additional risk metrics
            try:
                returns_series = result['returns']['returns']
                window = 36  # 3-year window
                
                rolling_ret = returns_series.rolling(window=window).mean() * 12
                rolling_vol = returns_series.rolling(window=window).std() * np.sqrt(12)
                
                summary.loc[strategy_name, 'Avg Rolling Return'] = rolling_ret.mean()
                summary.loc[strategy_name, 'Avg Rolling Vol'] = rolling_vol.mean()
                
                returns_array = returns_series.values
                summary.loc[strategy_name, 'Skewness'] = scipy.stats.skew(returns_array)
                summary.loc[strategy_name, 'Kurtosis'] = scipy.stats.kurtosis(returns_array)
                
                negative_returns = returns_array[returns_array < 0]
                summary.loc[strategy_name, 'Downside Vol'] = np.std(negative_returns) * np.sqrt(12) if len(negative_returns) > 0 else 0
                summary.loc[strategy_name, 'Max Monthly Loss'] = np.min(returns_array)
                
                # Drawdown statistics
                cum_returns = (1 + returns_series).cumprod()
                rolling_max = cum_returns.expanding().max()
                drawdowns = (cum_returns - rolling_max) / rolling_max
                
                summary.loc[strategy_name, 'Avg Drawdown'] = drawdowns.mean()
                summary.loc[strategy_name, 'Drawdown Duration'] = calculate_avg_drawdown_duration(drawdowns)
            except Exception as e:
                print(f"Error calculating risk metrics for {strategy_name}: {str(e)}")
                
        except Exception as e:
            print(f"Error processing strategy {strategy_name}: {str(e)}")
    
    return summary

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Set pandas display options for better output readability
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 3)
    
    try:
        print("Starting portfolio backtest...")
        
        # Run the backtest
        results = test_multi_asset_backtest()
        
        if results is not None:
            print("\nBacktest completed successfully!")
            print(f"\nResults saved in: {os.path.abspath('Output_Backtest')}")
            
            # Print summary of available files
            output_dir = 'Output_Backtest'
            files = os.listdir(output_dir)
            print("\nGenerated files:")
            for file in files:
                print(f"- {file}")
                
            # Print strategy names that were tested
            if 'results' in results:
                print("\nTested strategies:")
                for strategy in results['results'].keys():
                    print(f"- {strategy}")
                    
        else:
            print("\nBacktest failed to complete.")
            
    except Exception as e:
        print(f"\nError running backtest: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()