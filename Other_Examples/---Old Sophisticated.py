
class RobustBacktestOptimizer(RobustPortfolioOptimizer):
    """Enhanced class for performing vectorized backtesting of portfolio optimization strategies with robust features"""
    
    def __init__(
        self,
        returns: pd.DataFrame,
        lookback_window: int = 36,  # 3 years of monthly data
        rebalance_frequency: int = 3,  # Quarterly rebalancing
        estimation_method: str = 'robust',  # 'robust' or 'standard'
        use_cross_validation: bool = True,
        transaction_cost_model: str = 'proportional',  # 'proportional' or 'fixed+proportional'
        fixed_cost: float = 0.0,  # Fixed cost per trade
        benchmark_returns: Optional[pd.Series] = None,
        risk_estimator: str = 'semicovariance',  # 'empirical', 'semicovariance', or 'shrinkage'
        **kwargs
    ):
        """
        Initialize enhanced backtester with robust features
        
        Args:
            returns: DataFrame of asset returns
            lookback_window: Length of estimation window
            rebalance_frequency: Number of periods between rebalancing
            estimation_method: Method for estimating parameters
            use_cross_validation: Whether to use cross-validation for parameter selection
            transaction_cost_model: Type of transaction cost model
            fixed_cost: Fixed cost per trade
            benchmark_returns: Optional benchmark return series
            risk_estimator: Method for estimating risk
            **kwargs: Additional parameters passed to parent
        """
        super().__init__(returns=returns, **kwargs)
        
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        self.estimation_method = estimation_method
        self.use_cross_validation = use_cross_validation
        self.transaction_cost_model = transaction_cost_model
        self.fixed_cost = fixed_cost
        self.benchmark_returns = benchmark_returns
        self.risk_estimator = risk_estimator
        
        # Initialize parameter history
        self.parameter_history = pd.DataFrame()
        
    def _estimate_parameters(self, historical_returns: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Estimate parameters using selected method with robust features
        
        Args:
            historical_returns: Historical return data
            
        Returns:
            Dictionary of estimated parameters
        """
        if self.estimation_method == 'robust':
            # Use robust estimation methods
            mu = self._robust_mean_estimate(historical_returns)
            sigma = self._robust_covariance_estimate(historical_returns)
        else:
            # Standard estimation
            mu = historical_returns.mean().values
            sigma = historical_returns.cov().values
            
        if self.use_cross_validation:
            # Cross-validate epsilon parameter
            epsilon = self._cross_validate_epsilon(historical_returns)
        else:
            epsilon = self.epsilon
            
        return {
            'expected_returns': mu,
            'covariance': sigma,
            'epsilon': epsilon
        }
        
    def _robust_mean_estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute robust mean estimate using Huber estimator"""
        from scipy.stats import norm
        
        def huber_mean(x: np.ndarray, k: float = 1.345) -> float:
            """Huber mean estimator"""
            n = len(x)
            mu = np.median(x)
            mad = np.median(np.abs(x - mu))
            scale = mad / norm.ppf(0.75)
            
            while True:
                w = np.clip(k * scale / np.abs(x - mu), 0, 1)
                mu_new = np.sum(w * x) / np.sum(w)
                if np.abs(mu_new - mu) < 1e-6:
                    break
                mu = mu_new
            return mu
            
        return np.array([huber_mean(returns[col]) for col in returns.columns])
        
    def _robust_covariance_estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute robust covariance estimate based on selected method
        """
        if self.risk_estimator == 'semicovariance':
            # Compute semicovariance matrix
            return self._compute_semicovariance(returns)
            
        elif self.risk_estimator == 'shrinkage':
            # Ledoit-Wolf shrinkage estimator
            return self._compute_shrinkage_covariance(returns)
            
        else:
            # Default to empirical covariance
            return returns.cov().values
            
    def _compute_semicovariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute semicovariance matrix using negative returns"""
        negative_returns = returns.where(returns < 0, 0)
        return negative_returns.cov().values
        
    def _compute_shrinkage_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute shrinkage covariance matrix using Ledoit-Wolf method"""
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        return lw.fit(returns).covariance_
        
    def _cross_validate_epsilon(self, returns: pd.DataFrame) -> float:
        """
        Cross-validate epsilon parameter using rolling window validation
        """
        window = self.lookback_window // 2
        epsilon_grid = np.linspace(0, 2 * self.epsilon, 10)
        cv_scores = []
        
        for eps in epsilon_grid:
            score = self._compute_cv_score(returns, window, eps)
            cv_scores.append(score)
            
        return epsilon_grid[np.argmin(cv_scores)]
        
    def _compute_cv_score(self, returns: pd.DataFrame, window: int, epsilon: float) -> float:
        """Compute cross-validation score for given epsilon"""
        cv_returns = []
        
        for t in range(window, len(returns) - window):
            # Train on first half
            train_returns = returns.iloc[t-window:t]
            
            # Create temporary optimizer
            temp_optimizer = RobustPortfolioOptimizer(
                returns=train_returns,
                epsilon=epsilon,
                alpha=self.alpha,
                optimization_method=self.optimization_method
            )
            
            # Optimize portfolio
            try:
                result = temp_optimizer.optimize(
                    objective=ObjectiveFunction.GARLAPPI_ROBUST,
                    constraints=OptimizationConstraints(long_only=True)
                )
                
                # Compute out-of-sample return
                test_return = returns.iloc[t:t+window] @ result['weights']
                cv_returns.append(test_return.mean())
                
            except Exception:
                continue
                
        return -np.mean(cv_returns)  # Negative because we minimize

    def run_backtest(
        self,
        objective: ObjectiveFunction,
        constraints: OptimizationConstraints,
        current_weights: pd.Series,
        **kwargs
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Run enhanced vectorized backtest with robust features and fixed weight handling
        
        Args:
            objective: Portfolio objective function
            constraints: Portfolio constraints
            current_weights: Initial portfolio weights as pd.Series
            **kwargs: Additional optimization parameters
        """
        returns = self.returns
        dates = returns.index
        n_assets = len(returns.columns)
        
        # Initialize results containers with proper indexing
        portfolio_weights = pd.DataFrame(0.0, index=dates, columns=returns.columns)
        portfolio_returns = pd.Series(0.0, index=dates)
        metrics_history = pd.DataFrame(index=dates)
        parameters_history = pd.DataFrame(index=dates)
        realized_costs = pd.Series(0.0, index=dates)
        
        # Ensure initial weights are properly indexed
        if not isinstance(current_weights, pd.Series):
            current_weights = pd.Series(current_weights, index=returns.columns)
        
        # Set initial weights
        portfolio_weights.iloc[0] = current_weights
        
        print("Running backtest with robust features...")
        try:
            for t in tqdm(range(self.lookback_window, len(returns))):
                current_date = dates[t]
                
                # Check if rebalancing is needed
                if (t - self.lookback_window) % self.rebalance_frequency == 0:
                    # Get historical data
                    historical_returns = returns.iloc[t-self.lookback_window:t]
                    
                    try:
                        # Estimate parameters
                        params = self._estimate_parameters(historical_returns)
                        
                        # Store parameter estimates
                        parameters_history.loc[current_date] = {
                            'epsilon': params['epsilon'],
                            'mean_return': np.mean(params['expected_returns']),
                            'mean_vol': np.sqrt(np.mean(np.diag(params['covariance'])))
                        }
                        
                        # Create temporary optimizer
                        period_optimizer = RobustPortfolioOptimizer(
                            returns=historical_returns,
                            epsilon=params['epsilon'],
                            alpha=self.alpha,
                            optimization_method=self.optimization_method,
                            half_life=self.half_life,
                            risk_free_rate=self.risk_free_rate,
                            transaction_cost=self.transaction_cost
                        )
                        
                        # Optimize portfolio
                        result = period_optimizer.optimize(
                            objective=objective,
                            constraints=constraints,
                            current_weights=current_weights.values,  # Pass numpy array
                            **kwargs
                        )
                        
                        # Convert optimization result to Series with proper index
                        new_weights = pd.Series(result['weights'], index=returns.columns)
                        
                        # Calculate transaction costs
                        costs = self._calculate_transaction_costs(
                            current_weights.values, 
                            new_weights.values
                        )
                        realized_costs.loc[current_date] = costs
                        
                        # Update weights
                        current_weights = new_weights
                        
                        # Store metrics
                        metrics = {
                            key: value for key, value in result.items() 
                            if key != 'weights'
                        }
                        metrics['transaction_costs'] = costs
                        metrics_history.loc[current_date] = pd.Series(metrics)
                        
                    except Exception as e:
                        print(f"Optimization failed at {current_date}: {str(e)}")
                        # Keep current weights if optimization fails
                        metrics_history.loc[current_date] = np.nan
                        parameters_history.loc[current_date] = np.nan
                        realized_costs.loc[current_date] = 0
                
                # Store weights with proper indexing
                portfolio_weights.loc[current_date] = current_weights
                
                # Calculate portfolio return
                period_return = returns.loc[current_date]
                portfolio_returns.loc[current_date] = (
                    (period_return * current_weights).sum() - 
                    realized_costs.loc[current_date]
                )
                
        except KeyboardInterrupt:
            print("\nBacktest interrupted by user")
            
        # Clean up results
        portfolio_returns = portfolio_returns.fillna(0)
        portfolio_weights = portfolio_weights.fillna(method='ffill')
        metrics_history = metrics_history.fillna(method='ffill')
        parameters_history = parameters_history.fillna(method='ffill')
        realized_costs = realized_costs.fillna(0)
        
        # Calculate final metrics
        backtest_metrics = self._calculate_backtest_metrics(
            portfolio_returns=portfolio_returns,
            portfolio_weights=portfolio_weights,
            metrics_history=metrics_history,
            parameters_history=parameters_history,
            realized_costs=realized_costs
        )
        
        return {
            'returns': portfolio_returns,
            'weights': portfolio_weights,
            'metrics_history': metrics_history,
            'parameters_history': parameters_history,
            'realized_costs': realized_costs,
            'backtest_metrics': backtest_metrics
        }

    def _calculate_backtest_metrics(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights: pd.DataFrame,
        metrics_history: pd.DataFrame,
        parameters_history: pd.DataFrame,
        realized_costs: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive backtest performance metrics"""
        metrics = {}
        
        # Ensure we have numeric data
        portfolio_returns = portfolio_returns.astype(float)
        returns_array = portfolio_returns.values
        
        # Return metrics
        metrics['Total Return'] = float((1 + portfolio_returns).prod() - 1)
        metrics['Annualized Return'] = float((1 + metrics['Total Return']) ** (12 / len(portfolio_returns)) - 1)
        metrics['Volatility'] = float(portfolio_returns.std() * np.sqrt(12))
        
        # Handle case where volatility is zero
        if metrics['Volatility'] > 0:
            metrics['Sharpe Ratio'] = float((metrics['Annualized Return'] - self.risk_free_rate) / metrics['Volatility'])
        else:
            metrics['Sharpe Ratio'] = 0.0
        
        # Risk metrics
        metrics['Skewness'] = float(scipy.stats.skew(returns_array))
        metrics['Kurtosis'] = float(scipy.stats.kurtosis(returns_array))
        metrics['VaR_95'] = float(np.percentile(returns_array, 5))
        var_returns = returns_array[returns_array <= metrics['VaR_95']]
        metrics['CVaR_95'] = float(np.mean(var_returns) if len(var_returns) > 0 else metrics['VaR_95'])
        
        # Drawdown analysis
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns / rolling_max - 1)
        metrics['Maximum Drawdown'] = float(drawdowns.min())
        metrics['Average Drawdown'] = float(drawdowns.mean())
        
        # Portfolio turnover and costs
        weight_changes = portfolio_weights.diff().abs().sum(axis=1)
        metrics['Average Turnover'] = float(weight_changes.mean())
        metrics['Total Costs'] = float(realized_costs.sum())
        
        # Avoid division by zero for cost ratio
        if abs(metrics['Total Return']) > 1e-10:
            metrics['Cost Ratio'] = float(metrics['Total Costs'] / metrics['Total Return'])
        else:
            metrics['Cost Ratio'] = 0.0
        
        return metrics

    def _calculate_transaction_costs(self, old_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """Calculate transaction costs using specified model with proper array handling"""
        if self.transaction_cost_model == 'proportional':
            return np.sum(np.abs(new_weights - old_weights)) * self.transaction_cost
        else:
            # Fixed + proportional model
            n_trades = np.sum(np.abs(new_weights - old_weights) > 1e-6)
            prop_costs = np.sum(np.abs(new_weights - old_weights)) * self.transaction_cost
            return n_trades * self.fixed_cost + prop_costs
        
    def _calculate_avg_drawdown_duration(self, drawdowns: pd.Series) -> float:
        """Calculate average drawdown duration in periods"""
        is_drawdown = drawdowns < 0
        drawdown_starts = is_drawdown
        

    def _calculate_avg_drawdown_duration(self, drawdowns: pd.Series) -> float:
        """Calculate average drawdown duration in periods"""
        is_drawdown = drawdowns < 0
        drawdown_starts = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        drawdown_ends = (~is_drawdown) & is_drawdown.shift(1).fillna(False)
        
        # Find drawdown periods
        start_dates = drawdown_starts[drawdown_starts].index
        end_dates = drawdown_ends[drawdown_ends].index
        
        if len(start_dates) == 0 or len(end_dates) == 0:
            return 0
            
        # Calculate durations
        durations = []
        current_start = None
        
        for start in start_dates:
            if current_start is None:
                current_start = start
            ends = end_dates[end_dates > start]
            if len(ends) > 0:
                duration = (ends[0] - start).days / 30  # Convert to months
                durations.append(duration)
                current_start = None
                
        return np.mean(durations) if durations else 0

    def _calculate_benchmark_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate benchmark-relative performance metrics"""
        # Align series
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        port_returns = aligned_returns.iloc[:, 0]
        bench_returns = aligned_returns.iloc[:, 1]
        
        # Calculate tracking error
        tracking_diff = port_returns - bench_returns
        tracking_error = tracking_diff.std() * np.sqrt(12)
        
        # Calculate information ratio
        active_return = port_returns.mean() - bench_returns.mean()
        information_ratio = (active_return * 12) / tracking_error
        
        # Calculate beta
        covariance = np.cov(port_returns, bench_returns)[0, 1]
        variance = np.var(bench_returns)
        beta = covariance / variance
        
        # Calculate alpha (CAPM)
        rf_monthly = self.risk_free_rate / 12
        excess_return = port_returns.mean() - rf_monthly
        market_premium = bench_returns.mean() - rf_monthly
        alpha = excess_return - beta * market_premium
        alpha = alpha * 12  # Annualize
        
        # Calculate up/down capture
        up_months = bench_returns > 0
        down_months = bench_returns < 0
        
        up_capture = (port_returns[up_months].mean() / 
                     bench_returns[up_months].mean()) if up_months.any() else 0
        down_capture = (port_returns[down_months].mean() / 
                       bench_returns[down_months].mean()) if down_months.any() else 0
        
        return {
            'Tracking Error': tracking_error,
            'Information Ratio': information_ratio,
            'Beta': beta,
            'Alpha': alpha,
            'Up Capture': up_capture,
            'Down Capture': down_capture
        }
        
    def plot_backtest_results(self, results: Dict[str, Union[pd.Series, pd.DataFrame]]):
        """Create comprehensive visualization of backtest results with additional plots"""
        fig = plt.figure(figsize=(20, 25))
        gs = plt.GridSpec(5, 2)
        
        # 1. Cumulative Returns
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_cumulative_returns(results['returns'], ax1)
        
        # 2. Drawdowns
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_drawdown(results['returns'], ax2)
        
        # 3. Rolling Risk Metrics
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_rolling_risk_metrics(results['returns'], ax3)
        
        # 4. Weight Evolution
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_weight_evolution(results['weights'], ax4)
        
        # 5. Risk Contribution Evolution
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_risk_contribution(results['weights'], ax5)
        
        # 6. Parameter Stability
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_parameter_stability(results['parameters_history'], ax6)
        
        # 7. Transaction Costs Analysis
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_transaction_costs(results['realized_costs'], ax7)
        
        # 8. Performance Distribution
        ax8 = fig.add_subplot(gs[4, 0])
        self._plot_return_distribution(results['returns'], ax8)
        
        # 9. Risk-Return Scatter
        ax9 = fig.add_subplot(gs[4, 1])
        self._plot_risk_return_scatter(results['returns'], ax9)
        
        plt.tight_layout()
        plt.show()
        
    def _plot_cumulative_returns(self, returns: pd.Series, ax: plt.Axes):
        """Enhanced cumulative returns plot with benchmark comparison"""
        cumulative_returns = (1 + returns).cumprod()
        cumulative_returns.plot(ax=ax, label='Portfolio', linewidth=2)
        
        if self.benchmark_returns is not None:
            benchmark_cum_returns = (1 + self.benchmark_returns).cumprod()
            benchmark_cum_returns.plot(ax=ax, label='Benchmark', 
                                    linestyle='--', alpha=0.7)
            
            # Add relative strength line
            relative_strength = cumulative_returns / benchmark_cum_returns
            ax2 = ax.twinx()
            relative_strength.plot(ax=ax2, label='Relative Strength',
                                color='gray', alpha=0.5)
            ax2.set_ylabel('Relative Strength')
            
        ax.set_title('Cumulative Returns')
        ax.legend(loc='upper left')
        ax.grid(True)
        
    def _plot_parameter_stability(self, params: pd.DataFrame, ax: plt.Axes):
        """Plot evolution of estimated parameters"""
        if not params.empty:
            # Normalize parameters for comparison
            normalized_params = (params - params.mean()) / params.std()
            normalized_params.plot(ax=ax)
            ax.set_title('Parameter Stability (Normalized)')
            ax.legend()
            ax.grid(True)
            
    def _plot_transaction_costs(self, costs: pd.Series, ax: plt.Axes):
        """Plot transaction costs analysis"""
        # Plot cumulative costs
        cumulative_costs = costs.cumsum()
        cumulative_costs.plot(ax=ax, label='Cumulative Costs')
        
        # Add rolling average
        rolling_costs = costs.rolling(window=12).mean()
        rolling_costs.plot(ax=ax, label='12-Month Rolling Average',
                         linestyle='--', alpha=0.7)
        
        ax.set_title('Transaction Costs Analysis')
        ax.legend()
        ax.grid(True)
        
    def _plot_return_distribution(self, returns: pd.Series, ax: plt.Axes):
        """Plot return distribution with normal comparison"""
        from scipy import stats
        
        returns.hist(ax=ax, bins=50, density=True, alpha=0.7,
                    label='Actual Returns')
        
        # Fit normal distribution
        mu, sigma = stats.norm.fit(returns)
        x = np.linspace(returns.min(), returns.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2,
               label='Normal Distribution')
        
        # Add metrics
        ax.text(0.05, 0.95, f'Skewness: {stats.skew(returns):.2f}\n'
                           f'Kurtosis: {stats.kurtosis(returns):.2f}',
               transform=ax.transAxes, verticalalignment='top')
        
        ax.set_title('Return Distribution')
        ax.legend()
        ax.grid(True)
        
    def _plot_risk_return_scatter(self, returns: pd.Series, ax: plt.Axes):
        """Plot risk-return scatter with efficient frontier comparison"""
        # Calculate rolling returns and volatilities
        window = 12
        rolling_returns = returns.rolling(window).mean() * 12
        rolling_vols = returns.rolling(window).std() * np.sqrt(12)
        
        ax.scatter(rolling_vols, rolling_returns, alpha=0.5)
        
        # Add regression line
        z = np.polyfit(rolling_vols, rolling_returns, 1)
        p = np.poly1d(z)
        ax.plot(rolling_vols, p(rolling_vols), "r--", alpha=0.8)
        
        ax.set_xlabel('Risk (Annualized Volatility)')
        ax.set_ylabel('Return (Annualized)')
        ax.set_title('Risk-Return Characteristics')
        ax.grid(True)
        
    def save_backtest_results(self, results: Dict[str, Union[pd.Series, pd.DataFrame]], 
                            filename: str):
        """Save backtest results to file"""
        results_to_save = {
            'returns': results['returns'].to_frame('returns'),
            'weights': results['weights'],
            'metrics_history': results['metrics_history'],
            'parameters_history': results['parameters_history'],
            'realized_costs': results['realized_costs'].to_frame('costs'),
            'backtest_metrics': results['backtest_metrics'].to_frame('metrics')
        }
        
        with pd.ExcelWriter(filename) as writer:
            for sheet_name, data in results_to_save.items():
                data.to_excel(writer, sheet_name=sheet_name)
