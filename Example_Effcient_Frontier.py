import pandas as pd
import numpy as np
from PortOpt import *
from typing import Dict, List, Optional, Tuple, Union, Callable

################################################################################################################
#### Testing Efficient Frontier
################################################################################################################


# Loading test data 
def get_test_data():
    """Get stock data from Yahoo Finance for testing"""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Define stock universe by sector
    stocks = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL'],
        'Finance': ['JPM', 'BAC', 'GS'],
        'Healthcare': ['JNJ', 'PFE', 'UNH'],
        'Consumer': ['PG', 'KO', 'WMT']
    }
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6*365)  # 3 years of data
    
    # Download data
    prices = pd.DataFrame()
    failed_downloads = []
    
    print("Downloading stock data...")
    for sector, symbols in stocks.items():
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if len(data) > 0:
                    prices[symbol] = data['Adj Close']
                else:
                    failed_downloads.append(symbol)
            except Exception as e:
                print(f"Failed to download {symbol}: {e}")
                failed_downloads.append(symbol)
    
    if failed_downloads:
        print(f"Warning: Failed to download data for: {', '.join(failed_downloads)}")
    
    prices = prices.resample('M').ffill()
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Handle missing data and outliers
    returns = returns.fillna(method='ffill').fillna(method='bfill')
    
    # Remove extreme outliers (more than 5 standard deviations)
    for col in returns.columns:
        mean = returns[col].mean()
        std = returns[col].std()
        returns[col] = returns[col].clip(mean - 5*std, mean + 5*std)
    
    # Get benchmark data (S&P 500)
    try:
        benchmark = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        benchmark_returns = benchmark['Adj Close'].pct_change().dropna()
    except:
        print("Warning: Failed to download benchmark data")
        benchmark_returns = pd.Series(index=returns.index, data=0)  # Fallback to zeros
    
    print(f"Successfully downloaded data for {len(returns.columns)} stocks")
    print(f"Data shape: {returns.shape}")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    
    return returns, stocks, benchmark_returns

#### Efficient Frontier ####

    
def run_comprehensive_tests():
    """Run both general portfolio optimization and efficient frontier tests"""
    # Get test data
    returns, stocks, benchmark_returns = get_test_data()
    
    print("\nRunning Efficient Frontier Analysis...")
    frontier_results = run_efficient_frontier_analysis(returns)
    
    return frontier_results

def run_efficient_frontier_analysis(returns: pd.DataFrame):
    """Run comprehensive efficient frontier analysis"""
    print("\nInitializing Robust Efficient Frontier Calculator...")
    
    # Initialize frontier calculator with different uncertainty levels
    uncertainty_levels = [0.05, 0.10, 0.15]
    frontier_results = {}
    
    for uncertainty in uncertainty_levels:
        print(f"\nComputing Efficient Frontier with uncertainty = {uncertainty:.2f}")
        
        # Initialize calculator
        calculator = RobustEfficientFrontier(
            optimization_method=OptimizationMethod.SCIPY,
            returns=returns,
            uncertainty=uncertainty,
            risk_aversion=1.0,
            half_life=36,
            risk_free_rate=0.02,
            transaction_cost=0.001
        )
        
        # Define constraints
        constraints = OptimizationConstraints(
            long_only=True,
            box_constraints={i: (0.0, 0.3) for i in range(len(returns.columns))},  # Max 30% per asset
            group_constraints={  # Sector constraints
                f"Sector_{i}": GroupConstraint(
                    assets=list(range(i*3, (i+1)*3)),
                    bounds=(0.1, 0.4)  # 10-40% per sector
                ) for i in range(4)
            }
        )
        
        # Compute frontier
        results = calculator.compute_efficient_frontier(
            n_points=15,
            constraints=constraints
        )
        
        frontier_results[uncertainty] = results
        
        # Plot individual frontier
        calculator.plot_frontier(results)
        
    # Plot comparison of frontiers
    plot_frontier_comparison(frontier_results)
    
    return frontier_results

def plot_frontier_comparison(frontier_results: Dict[float, Dict[str, np.ndarray]]):
    """Plot comparison of efficient frontiers with different uncertainty levels"""
    plt.figure(figsize=(12, 8))
    
    for uncertainty, results in frontier_results.items():
        # Plot standard frontier
        plt.plot(results['risks'], results['returns'], 
                label=f'Expected Returns (μ={uncertainty:.2f})')
        
        # Plot worst-case frontier
        plt.plot(results['risks'], results['worst_case_returns'], '--',
                label=f'Worst-Case Returns (μ={uncertainty:.2f})')
        
        # Fill between expected and worst-case
        plt.fill_between(results['risks'], 
                        results['worst_case_returns'],
                        results['returns'], 
                        alpha=0.1)
    
    plt.xlabel('Portfolio Risk (Volatility)')
    plt.ylabel('Portfolio Return')
    plt.title('Comparison of Robust Efficient Frontiers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def print_frontier_metrics(frontier_results: Dict[float, Dict[str, np.ndarray]]):
    """Print summary metrics for each frontier"""
    print("\nEfficient Frontier Summary Metrics:")
    print("-" * 50)
    
    for uncertainty, results in frontier_results.items():
        print(f"\nUncertainty Level: {uncertainty:.2f}")
        print("-" * 30)
        
        # Calculate key metrics
        max_sharpe_idx = np.argmax(results['sharpe_ratios'])
        min_risk_idx = np.argmin(results['risks'])
        max_return_idx = np.argmax(results['returns'])
        
        # Print metrics for different portfolio types
        print("\nMaximum Sharpe Ratio Portfolio:")
        print(f"Return: {results['returns'][max_sharpe_idx]:.4%}")
        print(f"Risk: {results['risks'][max_sharpe_idx]:.4%}")
        print(f"Sharpe Ratio: {results['sharpe_ratios'][max_sharpe_idx]:.4f}")
        print(f"Worst-Case Return: {results['worst_case_returns'][max_sharpe_idx]:.4%}")
        
        print("\nMinimum Risk Portfolio:")
        print(f"Return: {results['returns'][min_risk_idx]:.4%}")
        print(f"Risk: {results['risks'][min_risk_idx]:.4%}")
        print(f"Sharpe Ratio: {results['sharpe_ratios'][min_risk_idx]:.4f}")
        print(f"Worst-Case Return: {results['worst_case_returns'][min_risk_idx]:.4%}")
        
        print("\nMaximum Return Portfolio:")
        print(f"Return: {results['returns'][max_return_idx]:.4%}")
        print(f"Risk: {results['risks'][max_return_idx]:.4%}")
        print(f"Sharpe Ratio: {results['sharpe_ratios'][max_return_idx]:.4f}")
        print(f"Worst-Case Return: {results['worst_case_returns'][max_return_idx]:.4%}")

def run_efficient_frontier_analysis(returns: pd.DataFrame):
    """Run comprehensive efficient frontier analysis with proper error handling"""
    print("\nInitializing Robust Efficient Frontier Calculator...")
    
    # Initialize frontier calculator with different uncertainty levels
    uncertainty_levels = [0.05, 0.10, 0.15]
    frontier_results = {}
    
    for uncertainty in uncertainty_levels:
        print(f"\nComputing Efficient Frontier with uncertainty = {uncertainty:.2f}")
        
        try:
            # Initialize calculator with explicit parameters
            calculator = RobustEfficientFrontier(
                returns=returns,
                uncertainty=uncertainty,
                risk_aversion=1.0,
                half_life=36,
                risk_free_rate=0.02,
                transaction_cost=0.001
            )
            
            # Define constraints
            base_constraints = OptimizationConstraints(
                long_only=True,
                box_constraints={i: (0.0, 0.3) for i in range(len(returns.columns))},
            )
            
            # Compute frontier with explicit parameters
            results = calculator.compute_efficient_frontier(
                n_points=30,
                constraints=base_constraints,
                risk_range=None,  # Will be computed automatically
                return_range=None  # Will be computed automatically
            )
            
            frontier_results[uncertainty] = results
            
            # Plot individual frontier
            calculator.plot_frontier(results)
            
        except Exception as e:
            print(f"Error computing frontier for uncertainty {uncertainty}: {str(e)}")
            continue
    
    if frontier_results:  # Only plot comparison if we have results
        plot_frontier_comparison(frontier_results)
    
    return frontier_results

if __name__ == "__main__":
    result = run_comprehensive_tests()