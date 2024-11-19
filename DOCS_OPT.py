"""
Portfolio Optimization Framework Documentation

This framework implements a comprehensive portfolio optimization system with robust optimization 
techniques. It consists of several key components:

Core Classes Hierarchy:
----------------------
1. PortfolioOptimizer (Base class)
   ├── RobustPortfolioOptimizer
   │   ├── RobustEfficientFrontier
   │   └── RobustBacktestOptimizer
   └── PortfolioDataHandler

Key Features:
------------
- Multiple optimization methods (SCIPY, CVXPY)
- Various objective functions (Minimum Variance, Mean-Variance, Robust optimization, etc.)
- Robust optimization with uncertainty parameters
- Efficient frontier computation
- Backtesting capabilities
- Comprehensive constraints handling
- Data preprocessing and cleaning
- Performance analytics and visualization

Main Components:
---------------
1. Data Handling:
   - Data preprocessing and cleaning
   - Missing value handling
   - Outlier detection and treatment
   - Time series alignment

2. Optimization:
   - Classical mean-variance optimization
   - Robust optimization with uncertainty
   - Multiple objective functions
   - Constraint handling
   - Risk management

3. Risk Management:
   - Robust estimation of parameters
   - Uncertainty handling
   - Transaction cost modeling
   - Risk contribution analysis

4. Performance Analysis:
   - Return metrics
   - Risk metrics
   - Portfolio analytics
   - Visualization tools

Usage Example:
-------------
```python
# Initialize optimizer
optimizer = RobustPortfolioOptimizer(
    returns=returns_data,
    expected_returns=expected_returns,
    epsilon=uncertainty_params,
    alpha=risk_aversion_params
)

# Define constraints
constraints = OptimizationConstraints(
    long_only=True,
    group_constraints={...},
    box_constraints={...}
)

# Optimize portfolio
result = optimizer.optimize(
    objective=ObjectiveFunction.GARLAPPI_ROBUST,
    constraints=constraints
)
```

Dependencies:
------------
- numpy: Numerical computations
- pandas: Data handling and manipulation
- scipy: Optimization routines
- cvxpy: Convex optimization
- matplotlib: Visualization
"""

# Core Classes Documentation

@dataclass
class GroupConstraint:
    """
    Defines group-level constraints for portfolio optimization.
    
    This class allows specification of minimum and maximum allocation bounds for groups 
    of assets (e.g., sectors, asset classes, regions).
    
    Attributes:
        assets (List[int]): List of asset indices that belong to the group.
        bounds (Tuple[float, float]): (min_weight, max_weight) for group allocation.
        
    Example:
        # Create constraint: Technology sector (assets 0,1,2) must be between 20-40%
        tech_constraint = GroupConstraint(
            assets=[0, 1, 2],
            bounds=(0.20, 0.40)
        )
        
    Notes:
        - Validates that asset indices are integers and bounds are valid (0 ≤ min ≤ max ≤ 1)
        - Used in OptimizationConstraints for group-level portfolio constraints
    """
    assets: List[int]
    bounds: Tuple[float, float]
    
    def __post_init__(self):
        """
        Validates constraint parameters upon initialization.
        
        Raises:
            ValueError: If assets are not integers or bounds are invalid
        """
        if not isinstance(self.assets, list) or not all(isinstance(i, int) for i in self.assets):
            raise ValueError("assets must be a list of integers")
        
        if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
            raise ValueError("bounds must be a tuple of (min_weight, max_weight)")
            
        min_weight, max_weight = self.bounds
        if not (0 <= min_weight <= max_weight <= 1):
            raise ValueError("bounds must satisfy 0 <= min_weight <= max_weight <= 1")

@dataclass
class OptimizationConstraints:
    """
    Comprehensive container for all portfolio optimization constraints.
    
    This class encapsulates all types of constraints that can be applied during 
    portfolio optimization, including group constraints, box constraints, and various 
    portfolio-level constraints.
    
    Attributes:
        group_constraints (Optional[Dict[str, GroupConstraint]]): 
            Named group-level constraints (e.g., sector, asset class constraints)
        box_constraints (Optional[Dict[int, Tuple[float, float]]]): 
            Individual asset min/max weights
        long_only (bool): Whether to enforce long-only constraint (default: True)
        max_turnover (Optional[float]): Maximum allowed portfolio turnover
        target_risk (Optional[float]): Target portfolio risk level
        target_return (Optional[float]): Target portfolio return
        max_tracking_error (Optional[float]): Maximum allowed tracking error
        benchmark_weights (Optional[np.ndarray]): Benchmark portfolio weights
        
    Example:
        constraints = OptimizationConstraints(
            long_only=True,
            group_constraints={
                'Technology': GroupConstraint(assets=[0,1,2], bounds=(0.1, 0.3)),
                'Financials': GroupConstraint(assets=[3,4,5], bounds=(0.2, 0.4))
            },
            box_constraints={
                0: (0.0, 0.1),  # Asset 0: max 10%
                1: (0.05, 0.15)  # Asset 1: 5-15%
            },
            max_turnover=0.2,  # 20% max turnover
            max_tracking_error=0.05  # 5% tracking error limit
        )
    
    Notes:
        - All constraints are optional
        - Constraints are enforced during optimization
        - Used by both SCIPY and CVXPY optimization methods
    """
    group_constraints: Optional[Dict[str, GroupConstraint]] = None
    box_constraints: Optional[Dict[int, Tuple[float, float]]] = None
    long_only: bool = True
    max_turnover: Optional[float] = None
    target_risk: Optional[float] = None
    target_return: Optional[float] = None
    max_tracking_error: Optional[float] = None
    benchmark_weights: Optional[np.ndarray] = None

"""
Optimization Methods and Objective Functions
------------------------------------------

The framework supports multiple optimization methods and objective functions through
the following key components:

1. OptimizationMethod: Enum defining available optimization solvers
2. ObjectiveFunction: Enum defining available objective functions
3. PortfolioObjective: Class implementing all objective functions
"""

class OptimizationMethod(Enum):
    """
    Enumeration of supported optimization methods.
    
    Available Methods:
        SCIPY: Uses scipy.optimize.minimize with SLSQP method
              - Advantages: Flexible, handles non-linear constraints
              - Best for: Smaller problems, custom objectives
        
        CVXPY: Uses CVXPY convex optimization
              - Advantages: Faster, more stable for convex problems
              - Best for: Large-scale problems, standard objectives
    """
    SCIPY = "scipy"
    CVXPY = "cvxpy"
    
class ObjectiveFunction(Enum):
    """
    Enumeration of available portfolio optimization objectives.
    
    Available Objectives:
        GARLAPPI_ROBUST: Robust optimization with parameter uncertainty
        MINIMUM_VARIANCE: Classic minimum variance portfolio
        MEAN_VARIANCE: Markowitz mean-variance optimization
        ROBUST_MEAN_VARIANCE: Robust version of mean-variance
        MAXIMUM_SHARPE: Maximum Sharpe ratio portfolio
        MAXIMUM_QUADRATIC_UTILITY: Maximize quadratic utility
        MINIMUM_TRACKING_ERROR: Minimize tracking error vs benchmark
        MAXIMUM_DIVERSIFICATION: Maximize portfolio diversification
        MINIMUM_CVAR: Minimize Conditional Value at Risk
        MEAN_CVAR: Mean-CVaR optimization
        RISK_PARITY: Risk parity portfolio
        EQUAL_RISK_CONTRIBUTION: Equal risk contribution portfolio
        HIERARCHICAL_RISK_PARITY: Hierarchical risk parity
    """
    GARLAPPI_ROBUST = "garlappi_robust"
    MINIMUM_VARIANCE = "minimum_variance"
    MEAN_VARIANCE = "mean_variance"
    ROBUST_MEAN_VARIANCE = "robust_mean_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    MAXIMUM_QUADRATIC_UTILITY = "maximum_quadratic_utility"
    MINIMUM_TRACKING_ERROR = "minimum_tracking_error"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    MINIMUM_CVAR = "minimum_cvar"
    MEAN_CVAR = "mean_cvar"
    RISK_PARITY = "risk_parity"
    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"

class PortfolioObjective:
    """
    Factory class implementing various portfolio optimization objective functions.
    
    This class provides static methods for all supported objective functions,
    each returning a callable that can be used by optimization algorithms.
    
    Key Components:
    --------------
    1. Estimation Error Handling:
       - Implements multiple methods for estimating parameter uncertainty
       - Supports asymptotic, Bayesian, and factor-based approaches
    
    2. Classic Objectives:
       - Minimum variance
       - Mean-variance
       - Maximum Sharpe ratio
       
    3. Robust Objectives:
       - Garlappi robust optimization
       - Robust mean-variance
       - Parameter uncertainty handling
       
    4. Risk-Based Objectives:
       - Risk parity
       - Equal risk contribution
       - Maximum diversification
       
    5. Downside Risk Objectives:
       - CVaR optimization
       - Mean-CVaR optimization
       
    Usage Example:
    -------------
    ```python
    # Create minimum variance objective
    min_var_obj = PortfolioObjective.minimum_variance(covariance_matrix)
    
    # Create robust optimization objective
    robust_obj = PortfolioObjective.garlappi_robust(
        returns=returns_data,
        epsilon=uncertainty_param,
        alpha=risk_aversion,
        omega_method='bayes'
    )
    ```
    """
    
    @staticmethod
    def __calculate_estimation_error_covariance(
        returns: np.ndarray,
        method: str = 'asymptotic'
    ) -> np.ndarray:
        """
        Calculate the covariance matrix of estimation errors (Omega matrix).
        
        This is a key component for robust optimization, implementing three methods:
        
        1. Asymptotic Method ('asymptotic'):
           - Classical approach using Sigma/T
           - Fastest but may underestimate uncertainty
           
        2. Bayesian Method ('bayes'):
           - Uses Bayesian posterior covariance
           - More conservative, better for small samples
           
        3. Factor Method ('factor'):
           - Uses PCA to separate systematic and idiosyncratic risk
           - Best for high-dimensional portfolios
           
        Args:
            returns: Historical returns matrix (T x N)
            method: Estimation method ('asymptotic', 'bayes', 'factor')
            
        Returns:
            Omega matrix (N x N) representing estimation uncertainty
            
        Notes:
            - Ensures positive definiteness of output matrix
            - Automatically handles numerical stability
        """
        T, N = returns.shape
        
        if method == 'asymptotic':
            sigma = np.cov(returns, rowvar=False)
            omega = sigma / T
            
        elif method == 'bayes':
            mu = np.mean(returns, axis=0)
            sigma = np.cov(returns, rowvar=False)
            
            # Calculate sample variance of mean estimator
            sample_var = np.zeros((N, N))
            for t in range(T):
                dev = (returns[t] - mu).reshape(-1, 1)
                sample_var += dev @ dev.T
            
            omega = sample_var / (T * (T - 1))
            
        elif method == 'factor':
            k = min(3, N - 1)  # Number of factors
            sigma = np.cov(returns, rowvar=False)
            
            # PCA decomposition
            eigenvals, eigenvecs = np.linalg.eigh(sigma)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:, idx]
            
            # Separate systematic and idiosyncratic components
            systematic_var = eigenvecs[:, :k] @ np.diag(eigenvals[:k]) @ eigenvecs[:, :k].T
            idiosyncratic_var = sigma - systematic_var
            
            omega = systematic_var / T + np.diag(np.diag(idiosyncratic_var)) / T
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Ensure numerical stability
        omega = (omega + omega.T) / 2
        min_eigenval = np.min(np.linalg.eigvals(omega))
        if min_eigenval < 0:
            omega += (-min_eigenval + 1e-8) * np.eye(N)
            
        return omega

"""
"""
Portfolio Objective Functions Implementation
-----------------------------------------

This section documents the implementation of each portfolio optimization objective function.
Each method is a static method returning a callable that can be used by the optimization
algorithms.

Key Categories:
1. Classical Objectives
2. Robust Objectives
3. Risk-Based Objectives
4. Tracking and Diversification Objectives
"""

class PortfolioObjective:
    @staticmethod
    def garlappi_robust(
        returns: np.ndarray, 
        epsilon: Union[float, np.ndarray], 
        alpha: np.ndarray, 
        omega_method: str = 'bayes', 
        omega: Optional[np.ndarray] = None
    ) -> callable:
        """
        Implements Garlappi et al. (2007) robust portfolio optimization.
        
        This is a sophisticated robust optimization approach that accounts for:
        - Parameter uncertainty in expected returns
        - Asset-specific uncertainty levels
        - Risk aversion preferences
        
        Mathematical Formulation:
        ------------------------
        Maximize: w'μ - √(εw'Ωw)w'Ωw/√(w'Ωw) - (λ/2)w'Σw
        where:
        - w: Portfolio weights
        - μ: Expected returns
        - Ω: Parameter uncertainty matrix
        - Σ: Covariance matrix
        - ε: Uncertainty parameter
        - λ: Risk aversion parameter
        
        Args:
            returns: Historical returns matrix (T x N)
            epsilon: Uncertainty parameter(s) - scalar or vector
            alpha: Risk aversion parameter(s) - vector
            omega_method: Method for uncertainty estimation ('bayes', 'asymptotic', 'factor')
            omega: Pre-computed uncertainty matrix (optional)
            
        Returns:
            Callable objective function for optimization
            
        Example:
            ```python
            obj_func = PortfolioObjective.garlappi_robust(
                returns=historical_returns,
                epsilon=0.1,
                alpha=np.ones(n_assets),
                omega_method='bayes'
            )
            ```
        """
        # Implementation details...

    @staticmethod
    def minimum_variance(Sigma: np.ndarray) -> callable:
        """
        Classical minimum variance optimization objective.
        
        Minimizes portfolio variance without return considerations:
        min w'Σw
        
        Args:
            Sigma: Covariance matrix
            
        Returns:
            Callable that computes portfolio variance
            
        Notes:
            - Simplest optimization objective
            - No estimation of expected returns needed
            - Often more stable than mean-variance
        """
        def objective(w: np.ndarray) -> float:
            return w.T @ Sigma @ w
        return objective

    @staticmethod
    def robust_mean_variance(
        mu: np.ndarray, 
        Sigma: np.ndarray, 
        epsilon: Union[float, np.ndarray], 
        kappa: float
    ) -> callable:
        """
        Robust version of mean-variance optimization.
        
        Incorporates uncertainty in expected returns through:
        - Asset-specific uncertainty parameters
        - Worst-case scenario analysis
        - Risk-return tradeoff adjustment
        
        Mathematical Formulation:
        ------------------------
        Minimize: -w'μ + (κ - ε'w)√(w'Σw)
        
        Args:
            mu: Expected returns
            Sigma: Covariance matrix
            epsilon: Uncertainty parameter(s)
            kappa: Risk aversion parameter
            
        Returns:
            Callable robust mean-variance objective
            
        Notes:
            - More conservative than standard mean-variance
            - Better handles parameter uncertainty
        """
        # Implementation details...

    @staticmethod
    def maximum_sharpe(
        mu: np.ndarray, 
        Sigma: np.ndarray, 
        rf_rate: float = 0.0
    ) -> callable:
        """
        Maximum Sharpe ratio optimization objective.
        
        Maximizes the portfolio Sharpe ratio:
        max (w'μ - rf) / √(w'Σw)
        
        Args:
            mu: Expected returns
            Sigma: Covariance matrix
            rf_rate: Risk-free rate
            
        Returns:
            Callable that computes negative Sharpe ratio
            
        Notes:
            - Non-convex optimization
            - May have multiple local optima
            - Scale-invariant to position sizing
        """
        # Implementation details...

    @staticmethod
    def maximum_diversification(
        Sigma: np.ndarray, 
        asset_stdevs: Optional[np.ndarray] = None
    ) -> callable:
        """
        Maximum diversification ratio optimization.
        
        Maximizes the ratio of weighted assets' stand-alone risks
        to portfolio risk:
        max (w'σ) / √(w'Σw)
        
        Args:
            Sigma: Covariance matrix
            asset_stdevs: Individual asset standard deviations
            
        Returns:
            Callable that computes negative diversification ratio
            
        Notes:
            - Focuses on risk diversification
            - Independent of expected returns
            - Good for risk-focused allocation
        """
        # Implementation details...

    @staticmethod
    def minimum_cvar(
        returns: np.ndarray, 
        alpha: float = 0.05, 
        scenarios: Optional[np.ndarray] = None
    ) -> callable:
        """
        Conditional Value at Risk (CVaR) minimization.
        
        Minimizes the expected shortfall beyond VaR:
        min E[X | X ≤ VaRα]
        
        Args:
            returns: Historical returns
            alpha: Confidence level (e.g., 0.05 for 95% CVaR)
            scenarios: Additional stress test scenarios
            
        Returns:
            Callable that computes CVaR
            
        Notes:
            - More stable than VaR optimization
            - Accounts for tail risk
            - Can incorporate stress scenarios
        """
        # Implementation details...

    @staticmethod
    def risk_parity(Sigma: np.ndarray) -> callable:
        """
        Risk parity portfolio optimization.
        
        Equalizes risk contribution from each asset:
        min Σ(RCi - RCj)²
        
        Args:
            Sigma: Covariance matrix
            
        Returns:
            Callable that computes risk parity objective
            
        Notes:
            - Focus on risk allocation
            - More stable than return-based methods
            - Popular in practice
        """
        # Implementation details...

Note:
-----
Each objective function is designed to work with both SCIPY and CVXPY optimizers.
The implementation details vary based on the mathematical properties of each objective
and the requirements of the optimization methods.
"""

"""
Portfolio Optimizer Base Class Documentation
-----------------------------------------

The PortfolioOptimizer class serves as the foundation for the entire optimization framework.
It implements core optimization functionality, handles different optimization methods,
and provides the base interface for more specialized optimizers.

Class Hierarchy:
---------------
PortfolioOptimizer
├── Basic optimization methods (SCIPY, CVXPY)
├── Constraint handling
├── Risk and return calculations
└── Performance metrics

Key Features:
------------
1. Multiple optimization methods support
2. Comprehensive constraint handling
3. Flexible objective function interface
4. Risk and return metrics calculation
5. Transaction cost modeling
6. Performance analytics
"""

class PortfolioOptimizer:
    """
    Base class for portfolio optimization.
    
    This class provides the core functionality for portfolio optimization,
    including multiple optimization methods, constraint handling, and
    performance analytics.
    
    Attributes:
        returns (pd.DataFrame): Historical returns data
        expected_returns (np.ndarray): Expected returns vector
        optimization_method (OptimizationMethod): SCIPY or CVXPY
        half_life (int): Half-life for exponential weighting
        risk_free_rate (float): Risk-free rate for Sharpe calculations
        transaction_cost (float): Transaction cost parameter
        covariance (np.ndarray): Covariance matrix
        objective_functions (PortfolioObjective): Objective function factory
        
    Key Methods:
        optimize: Main optimization method
        _optimize_scipy: SCIPY-based optimization
        _optimize_cvxpy: CVXPY-based optimization
        _calculate_metrics: Calculate portfolio metrics
        
    Example:
        ```python
        optimizer = PortfolioOptimizer(
            returns=historical_returns,
            expected_returns=forecasted_returns,
            optimization_method=OptimizationMethod.CVXPY,
            risk_free_rate=0.02,
            transaction_cost=0.001
        )
        
        result = optimizer.optimize(
            objective=ObjectiveFunction.MEAN_VARIANCE,
            constraints=constraints,
            current_weights=initial_weights
        )
        ```
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[np.ndarray] = None,
        optimization_method: OptimizationMethod = OptimizationMethod.SCIPY,
        half_life: int = 36,
        risk_free_rate: float = 0.0,
        transaction_cost: float = 0.001
    ):
        """
        Initialize Portfolio Optimizer.
        
        Args:
            returns: Historical returns data
            expected_returns: Expected returns (optional)
            optimization_method: SCIPY or CVXPY
            half_life: Half-life for exponential weighting
            risk_free_rate: Risk-free rate
            transaction_cost: Transaction cost parameter
            
        Notes:
            - If expected_returns not provided, uses historical mean
            - Uses exponentially weighted covariance by default
            - Initializes with reasonable default parameters
        """
        self.returns = returns
        self.optimization_method = optimization_method
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        
        # Compute exponentially weighted covariance matrix
        self.covariance = self._compute_ewm_covariance(half_life)
        
        # Use provided expected returns or compute from historical data
        self.expected_returns = (
            expected_returns if expected_returns is not None 
            else self._compute_expected_returns()
        )
        
        # Initialize portfolio objective functions
        self.objective_functions = PortfolioObjective()

    def optimize(
        self,
        objective: ObjectiveFunction,
        constraints: OptimizationConstraints,
        current_weights: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Main optimization method with automatic fallback mechanisms.
        
        This method implements a sophisticated optimization approach with multiple
        fallback strategies:
        1. Try original method with original constraints
        2. Try alternative method if first attempt fails
        3. Try with relaxed constraints
        4. Try with minimal constraints as last resort
        
        Args:
            objective: Optimization objective function
            constraints: Optimization constraints
            current_weights: Current portfolio weights
            **kwargs: Additional parameters for specific objectives
            
        Returns:
            Dictionary containing:
            - weights: Optimal portfolio weights
            - return: Expected portfolio return
            - risk: Portfolio risk
            - sharpe_ratio: Sharpe ratio
            - turnover: Portfolio turnover
            - transaction_costs: Transaction costs
            - net_return: Return after costs
            
        Raises:
            ValueError: If all optimization attempts fail
            
        Example:
            ```python
            result = optimizer.optimize(
                objective=ObjectiveFunction.MINIMUM_VARIANCE,
                constraints=OptimizationConstraints(
                    long_only=True,
                    max_turnover=0.2
                )
            )
            weights = result['weights']
            exp_return = result['return']
            ```
        """
        # Implementation details...

    def _optimize_scipy(
        self,
        objective: ObjectiveFunction,
        constraints: OptimizationConstraints,
        current_weights: np.ndarray,
        **kwargs
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize using SCIPY's minimize function.
        
        Implements optimization using SCIPY's SLSQP method with:
        - Comprehensive constraint handling
        - Numerical stability improvements
        - Error handling and validation
        
        Args:
            objective: Optimization objective
            constraints: Optimization constraints
            current_weights: Starting weights
            **kwargs: Additional parameters
            
        Returns:
            Optimization results dictionary
            
        Notes:
            - Uses SLSQP method for constraint handling
            - Automatically handles constraint conversion
            - Includes numerical stability checks
        """
        # Implementation details...

    def _optimize_cvxpy(
        self,
        objective: ObjectiveFunction,
        constraints: OptimizationConstraints,
        current_weights: np.ndarray,
        **kwargs
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize using CVXPY convex optimization.
        
        Implements optimization using CVXPY with:
        - Efficient convex problem formulation
        - Robust constraint handling
        - Advanced solver options
        
        Args:
            objective: Optimization objective
            constraints: Optimization constraints
            current_weights: Starting weights
            **kwargs: Additional parameters
            
        Returns:
            Optimization results dictionary
            
        Notes:
            - More efficient for convex problems
            - Better numerical stability
            - Handles large-scale problems better
        """
        # Implementation details...

"""
Robust Portfolio Optimizer Documentation
-------------------------------------

The RobustPortfolioOptimizer extends the base PortfolioOptimizer class with robust
optimization capabilities to handle parameter uncertainty and estimation error.

Class Hierarchy:
---------------
RobustPortfolioOptimizer (extends PortfolioOptimizer)
├── Uncertainty handling
├── Robust optimization methods
├── Parameter estimation
└── Additional risk metrics

Key Features:
------------
1. Parameter uncertainty modeling
2. Multiple estimation methods
3. Robust optimization objectives
4. Enhanced risk metrics
5. Adaptive parameter estimation
"""

class RobustPortfolioOptimizer(PortfolioOptimizer):
    """
    Enhanced portfolio optimizer with robust optimization capabilities.
    
    This class extends the base optimizer by incorporating:
    - Parameter uncertainty through epsilon parameters
    - Risk aversion through alpha parameters
    - Robust estimation methods
    - Enhanced risk metrics
    
    Attributes:
        All from PortfolioOptimizer, plus:
        epsilon (pd.DataFrame): Uncertainty parameters
        alpha (pd.DataFrame): Risk aversion parameters
        omega (np.ndarray): Estimation error covariance
        omega_method (str): Estimation method
        correlation (pd.DataFrame): Asset correlation matrix
        statistics (pd.DataFrame): Additional statistical metrics
        rolling_vol (pd.DataFrame): Rolling volatility estimates
        
    Example:
        ```python
        optimizer = RobustPortfolioOptimizer(
            returns=returns_data,
            expected_returns=expected_returns,
            epsilon=uncertainty_params,
            alpha=risk_aversion,
            omega_method='bayes',
            half_life=36
        )
        
        result = optimizer.optimize(
            objective=ObjectiveFunction.GARLAPPI_ROBUST,
            constraints=constraints
        )
        ```
    """
    
    def __init__(
        self, 
        returns: pd.DataFrame,
        expected_returns: Optional[pd.DataFrame] = None,
        epsilon: Optional[pd.DataFrame] = None,
        alpha: Optional[pd.DataFrame] = None,
        omega_method: str = 'bayes',
        optimization_method: OptimizationMethod = OptimizationMethod.SCIPY,
        half_life: int = 36,
        risk_free_rate: float = 0.01,
        transaction_cost: float = 0.001,
        min_history: int = 24
    ):
        """
        Initialize Robust Portfolio Optimizer.
        
        Args:
            returns: Historical returns data
            expected_returns: Expected returns (optional)
            epsilon: Uncertainty parameters (optional)
            alpha: Risk aversion parameters (optional)
            omega_method: Estimation method ('bayes', 'asymptotic', 'factor')
            optimization_method: SCIPY or CVXPY
            half_life: Half-life for exponential weighting
            risk_free_rate: Risk-free rate
            transaction_cost: Transaction cost parameter
            min_history: Minimum required history
            
        Notes:
            - Processes data using PortfolioDataHandler
            - Initializes default parameters if not provided
            - Computes estimation error covariance
        """
        # Implementation details...

    def optimize(
        self, 
        objective: Optional[ObjectiveFunction] = None,
        constraints: Optional[OptimizationConstraints] = None,
        current_weights: Optional[np.ndarray] = None, 
        **kwargs
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Enhanced optimization method with robust defaults.
        
        Features:
        - Default to Garlappi robust optimization
        - Automatic parameter handling
        - Multiple fallback strategies
        
        Args:
            objective: Optimization objective (defaults to GARLAPPI_ROBUST)
            constraints: Optimization constraints
            current_weights: Current portfolio weights
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing optimization results
            
        Notes:
            - Uses SCIPY for Garlappi optimization
            - Includes multiple fallback strategies
            - Handles parameter uncertainty
        """
        # Implementation details...

    def calculate_robust_metrics(
        self, 
        weights: np.ndarray, 
        alpha: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate additional robust performance metrics.
        
        Computes enhanced metrics including:
        - Worst-case return
        - Diversification ratio
        - Portfolio concentration
        - Estimation uncertainty
        - Risk contributions
        
        Args:
            weights: Portfolio weights
            alpha: Risk aversion parameters (optional)
            
        Returns:
            Dictionary containing:
            - worst_case_return: Worst-case scenario return
            - diversification_ratio: Portfolio diversification measure
            - herfindahl_index: Concentration measure
            - effective_n: Effective number of assets
            - estimation_uncertainty: Parameter uncertainty measure
            - robust_sharpe: Risk-adjusted return with uncertainty
            - risk_contributions: Asset risk contributions
            
        Example:
            ```python
            metrics = optimizer.calculate_robust_metrics(
                weights=optimal_weights
            )
            worst_case = metrics['worst_case_return']
            div_ratio = metrics['diversification_ratio']
            ```
        """
        # Implementation details...

    def _calculate_estimation_error_covariance(
        self, 
        returns: np.ndarray, 
        method: str = 'bayes'
    ) -> np.ndarray:
        """
        Calculate estimation error covariance matrix.
        
        Implements multiple estimation methods:
        1. Bayesian: Using posterior distribution
        2. Asymptotic: Using classical approach
        3. Factor: Using PCA decomposition
        
        Args:
            returns: Historical returns
            method: Estimation method
            
        Returns:
            Estimation error covariance matrix
            
        Notes:
            - Ensures positive definiteness
            - Handles numerical stability
            - Validates input parameters
        """
        # Implementation details...

    def _relax_constraints(
        self, 
        constraints: OptimizationConstraints
    ) -> OptimizationConstraints:
        """
        Create relaxed version of optimization constraints.
        
        Implements gradual constraint relaxation:
        1. Remove target constraints
        2. Relax box constraints
        3. Relax group constraints
        
        Args:
            constraints: Original constraints
            
        Returns:
            Relaxed constraints
            
        Notes:
            - Maintains feasibility
            - Preserves key constraints
            - Gradual relaxation approach
        """
        # Implementation details...

    def _garlappi_robust_fallback(
        self, 
        current_weights: np.ndarray
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Fallback optimization with minimal constraints.
        
        Implements simplified robust optimization:
        1. Uses only basic constraints
        2. Focuses on numerical stability
        3. Ensures feasible solution
        
        Args:
            current_weights: Starting weights
            
        Returns:
            Optimization results
            
        Notes:
            - Last resort optimization
            - Uses minimal constraints
            - Ensures solution existence
        """
        # Implementation details...

"""
Robust Backtesting Framework Documentation
---------------------------------------

The RobustBacktestOptimizer extends RobustPortfolioOptimizer to provide comprehensive
backtesting capabilities with robust optimization methods and advanced analytics.

Class Hierarchy:
---------------
RobustBacktestOptimizer (extends RobustPortfolioOptimizer)
├── Rolling window optimization
├── Out-of-sample testing
├── Performance analytics
└── Visualization tools

Key Features:
------------
1. Rolling window analysis
2. Transaction cost modeling
3. Performance measurement
4. Risk analytics
5. Visualization tools
"""

class RobustBacktestOptimizer(RobustPortfolioOptimizer):
    """
    Comprehensive backtesting framework with robust optimization.
    
    This class provides tools for:
    - Rolling window portfolio optimization
    - Out-of-sample performance testing
    - Transaction cost analysis
    - Performance and risk analytics
    - Result visualization
    
    Attributes:
        All from RobustPortfolioOptimizer, plus:
        lookback_window (int): Rolling window size
        rebalance_frequency (int): Rebalancing period
        estimation_method (str): Parameter estimation method
        benchmark_returns (pd.DataFrame): Benchmark returns
        out_of_sample (bool): Out-of-sample testing flag
        
    Example:
        ```python
        backtest = RobustBacktestOptimizer(
            returns=returns_data,
            expected_returns=expected_returns,
            epsilon=uncertainty_params,
            lookback_window=36,
            rebalance_frequency=3
        )
        
        results = backtest.run_backtest(
            objective=ObjectiveFunction.GARLAPPI_ROBUST,
            constraints=constraints
        )
        ```
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[pd.DataFrame] = None,
        epsilon: Optional[pd.DataFrame] = None,
        alpha: Optional[pd.DataFrame] = None,
        lookback_window: int = 36,
        rebalance_frequency: int = 3,
        estimation_method: str = 'robust',
        transaction_cost: float = 0.001,
        benchmark_returns: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
        min_history: int = 24,
        out_of_sample: bool = False,
        **kwargs
    ):
        """
        Initialize backtesting framework.
        
        Args:
            returns: Historical returns data
            expected_returns: Expected returns (optional)
            epsilon: Uncertainty parameters (optional)
            alpha: Risk aversion parameters (optional)
            lookback_window: Rolling window size
            rebalance_frequency: Rebalancing period
            estimation_method: Parameter estimation method
            transaction_cost: Transaction cost parameter
            benchmark_returns: Benchmark returns (optional)
            risk_free_rate: Risk-free rate
            min_history: Minimum required history
            out_of_sample: Out-of-sample testing flag
            **kwargs: Additional parameters
            
        Notes:
            - Handles data alignment
            - Validates input parameters
            - Initializes backtesting framework
        """
        # Implementation details...

    def run_backtest(
        self,
        objective: ObjectiveFunction,
        constraints: OptimizationConstraints,
        initial_weights: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Execute backtesting analysis.
        
        Implements comprehensive backtesting:
        1. Rolling window optimization
        2. Performance tracking
        3. Risk measurement
        4. Transaction cost analysis
        
        Args:
            objective: Portfolio optimization objective
            constraints: Portfolio constraints
            initial_weights: Starting portfolio weights
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing:
            - returns: Portfolio returns series
            - weights: Portfolio weights history
            - metrics_history: Performance metrics
            - realized_costs: Transaction costs
            - epsilon_history: Uncertainty parameters
            - backtest_metrics: Aggregate metrics
            
        Example:
            ```python
            results = backtest.run_backtest(
                objective=ObjectiveFunction.MEAN_VARIANCE,
                constraints=constraints
            )
            
            returns = results['returns']
            metrics = results['backtest_metrics']
            ```
        """
        # Implementation details...

    def _calculate_backtest_metrics(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights: pd.DataFrame,
        realized_costs: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate comprehensive backtest performance metrics.
        
        Computes key performance indicators:
        1. Return metrics
        2. Risk metrics
        3. Risk-adjusted returns
        4. Cost analysis
        
        Args:
            portfolio_returns: Historical returns
            portfolio_weights: Portfolio weights history
            realized_costs: Transaction costs
            
        Returns:
            Dictionary containing:
            - Total Return
            - Annualized Return
            - Volatility
            - Sharpe Ratio
            - Maximum Drawdown
            - Average Turnover
            - Total Costs
            
        Notes:
            - Handles missing data
            - Ensures proper scaling
            - Validates calculations
        """
        # Implementation details...

    def _calculate_benchmark_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate benchmark-relative performance metrics.
        
        Computes benchmark comparison metrics:
        1. Tracking error
        2. Information ratio
        3. Beta and Alpha
        4. Up/Down capture
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Dictionary containing:
            - Tracking Error
            - Information Ratio
            - Beta
            - Alpha
            - Up/Down Capture
            
        Notes:
            - Aligns time series
            - Handles missing data
            - Annualizes metrics
        """
        # Implementation details...

    def plot_backtest_results(
        self, 
        results: Dict[str, Union[pd.Series, pd.DataFrame]]
    ):
        """
        Create comprehensive visualization of backtest results.
        
        Generates multiple plots:
        1. Cumulative returns
        2. Rolling metrics
        3. Risk analysis
        4. Portfolio composition
        5. Transaction costs
        
        Args:
            results: Backtest results dictionary
            
        Notes:
            - Creates multiple subplots
            - Includes interactive elements
            - Saves to output directory
            
        Example:
            ```python
            backtest.plot_backtest_results(results)
            ```
        """
        # Implementation details...

    def save_backtest_results(
        self, 
        results: Dict[str, Union[pd.Series, pd.DataFrame]], 
        filename: str
    ):
        """
        Save backtest results to file.
        
        Features:
        - Multiple format support
        - Timezone handling
        - Data validation
        
        Args:
            results: Backtest results
            filename: Output filename
            
        Notes:
            - Handles time zones
            - Validates data
            - Creates directories
        """
        # Implementation details...
        

"""
Utility Functions, Data Handling, and Visualization Tools
-----------------------------------------------------

This section covers the supporting infrastructure for the portfolio optimization framework,
including data preprocessing, analysis tools, and visualization capabilities.

Components:
----------
1. Data Handler
2. Analysis Tools
3. Visualization Functions
4. Helper Utilities
"""

class PortfolioDataHandler:
    """
    Comprehensive data preprocessing and validation framework.
    
    This class handles:
    - Data cleaning and validation
    - Missing value treatment
    - Outlier detection
    - Time series alignment
    - Statistical calculations
    
    Example:
        ```python
        handler = PortfolioDataHandler(min_history=24)
        processed_data = handler.process_data(
            returns=returns_data,
            expected_returns=forecasts,
            benchmark_returns=benchmark
        )
        ```
    """
    
    def process_data(
        self,
        returns: pd.DataFrame,
        benchmark_returns: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.DataFrame] = None,
        epsilon: Optional[pd.DataFrame] = None,
        alpha: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Process and validate all input data.
        
        Comprehensive data processing pipeline:
        1. Clean and validate returns
        2. Handle missing data
        3. Remove outliers
        4. Calculate metrics
        5. Align all inputs
        
        Args:
            returns: Historical returns
            benchmark_returns: Benchmark data
            expected_returns: Expected returns
            epsilon: Uncertainty parameters
            alpha: Risk aversion parameters
            
        Returns:
            Dictionary containing processed data
            
        Raises:
            ValueError: If data validation fails
        """
        processed_data = {}
        
        # Process core returns data
        clean_returns = self._clean_returns(returns)
        if len(clean_returns) < self.min_history:
            raise ValueError(f"Insufficient data: {len(clean_returns)} periods < {self.min_history} minimum")
        
        clean_returns = self._handle_missing_data(clean_returns)
        clean_returns = self._remove_outliers(clean_returns)
        processed_data['returns'] = clean_returns
        
        # Process additional inputs
        if benchmark_returns is not None:
            processed_data['benchmark_returns'] = self._process_benchmark(
                clean_returns, 
                benchmark_returns
            )
        
        if expected_returns is not None:
            processed_data['expected_returns'] = self._validate_data_alignment(
                clean_returns, 
                expected_returns
            )
        
        # Process parameters
        if epsilon is not None:
            processed_data['epsilon'] = self._validate_data_alignment(
                clean_returns, 
                epsilon
            )
        
        if alpha is not None:
            processed_data['alpha'] = self._validate_data_alignment(
                clean_returns, 
                alpha
            )
        
        # Calculate additional metrics
        processed_data.update(self._calculate_metrics(clean_returns))
        
        return processed_data

    def _remove_outliers(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Remove statistical outliers from return data.
        
        Methods:
        1. Z-score based detection
        2. Winsorization
        3. Median replacement
        
        Args:
            returns: Returns data
            
        Returns:
            Cleaned returns
            
        Notes:
            - Uses 3-sigma threshold
            - Replaces with median
            - Preserves time structure
        """
        clean_returns = returns.copy()
        
        for column in returns.columns:
            series = returns[column]
            z_scores = np.abs((series - series.mean()) / series.std())
            clean_returns.loc[z_scores > 3, column] = series.median()
            
        return clean_returns

    def _calculate_metrics(self, returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate comprehensive statistical metrics.
        
        Computes:
        1. Correlation matrix
        2. Rolling metrics
        3. Return statistics
        4. Risk measures
        
        Args:
            returns: Returns data
            
        Returns:
            Dictionary of metrics
            
        Notes:
            - Handles rolling windows
            - Computes annualized metrics
            - Includes higher moments
        """
        metrics = {}
        
        # Correlation matrix
        metrics['correlation'] = returns.corr()
        
        # Rolling metrics
        window = min(12, len(returns) // 4)
        metrics['rolling_vol'] = returns.rolling(window=window).std() * np.sqrt(12)
        metrics['rolling_corr'] = returns.rolling(window=window).corr()
        
        # Return statistics
        stats = pd.DataFrame(index=returns.columns)
        stats['annualized_return'] = (1 + returns.mean()) ** 12 - 1
        stats['annualized_vol'] = returns.std() * np.sqrt(12)
        stats['skewness'] = returns.skew()
        stats['kurtosis'] = returns.kurtosis()
        metrics['statistics'] = stats
        
        return metrics

def create_analysis_plots(
    frontier_results: dict, 
    data: dict,
    output_dir: str = 'output'
):
    """
    Create comprehensive visualization suite.
    
    Generates:
    1. Efficient frontier plot
    2. Asset allocation evolution
    3. Risk analysis
    4. Performance metrics
    
    Args:
        frontier_results: Optimization results
        data: Market data
        output_dir: Output directory
        
    Example:
        ```python
        create_analysis_plots(
            frontier_results=optimization_results,
            data=market_data,
            output_dir='analysis_output'
        )
        ```
    """
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(2, 2)
    
    # Create individual plots
    _plot_efficient_frontier(frontier_results, fig.add_subplot(gs[0, 0]))
    _plot_allocation_evolution(frontier_results, data, fig.add_subplot(gs[0, 1]))
    _plot_risk_analysis(frontier_results, fig.add_subplot(gs[1, 0]))
    _plot_metrics_comparison(frontier_results, fig.add_subplot(gs[1, 1]))
    
    plt.tight_layout()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_analysis_report(
    results: Dict[str, Union[pd.DataFrame, Dict]],
    output_file: str = 'report.html'
) -> str:
    """
    Generate comprehensive HTML analysis report.
    
    Includes:
    1. Performance summary
    2. Risk metrics
    3. Portfolio composition
    4. Interactive charts
    
    Args:
        results: Analysis results
        output_file: Output filename
        
    Returns:
        Path to generated report
        
    Example:
        ```python
        report_path = generate_analysis_report(
            results=analysis_results,
            output_file='portfolio_analysis.html'
        )
        ```
    """
    # Report generation implementation...
    
"""
Complete Usage Examples and Best Practices
---------------------------------------

This section provides comprehensive examples and best practices for using
the portfolio optimization framework in different scenarios.

Contents:
1. Basic Usage Examples
2. Advanced Optimization Scenarios
3. Backtesting Examples
4. Best Practices
5. Common Pitfalls
"""

def basic_optimization_example():
    """
    Basic portfolio optimization example with minimum variance objective.
    
    Demonstrates:
    - Data preparation
    - Basic constraints
    - Simple optimization
    - Results analysis
    """
    # Generate or load data
    returns_data = generate_synthetic_multi_asset_data(
        n_years=5,
        freq='M',
        seed=42
    )
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(
        returns=returns_data['returns'],
        risk_free_rate=0.02,
        transaction_cost=0.001
    )
    
    # Define constraints
    constraints = OptimizationConstraints(
        long_only=True,
        box_constraints={
            i: (0.0, 0.2)  # Maximum 20% per asset
            for i in range(len(returns_data['returns'].columns))
        }
    )
    
    # Run optimization
    result = optimizer.optimize(
        objective=ObjectiveFunction.MINIMUM_VARIANCE,
        constraints=constraints
    )
    
    # Analyze results
    print("\nOptimization Results:")
    print(f"Expected Return: {result['return']:.2%}")
    print(f"Portfolio Risk: {result['risk']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    
    return result

def robust_optimization_example():
    """
    Advanced example using robust optimization with uncertainty.
    
    Demonstrates:
    - Uncertainty parameters
    - Robust optimization
    - Advanced constraints
    - Risk analysis
    """
    # Prepare data
    market_data = generate_synthetic_multi_asset_data(
        n_years=5,
        freq='M',
        seed=42
    )
    
    # Generate uncertainty parameters
    epsilon = generate_asset_specific_epsilon(
        returns=market_data['returns'],
        asset_mapping=market_data['asset_mapping'],
        base_epsilon=0.1
    )
    
    # Initialize robust optimizer
    optimizer = RobustPortfolioOptimizer(
        returns=market_data['returns'],
        epsilon=epsilon,
        omega_method='bayes',
        half_life=36
    )
    
    # Define comprehensive constraints
    constraints = OptimizationConstraints(
        long_only=True,
        box_constraints={
            i: (0.0, 0.25)  # Maximum 25% per asset
            for i in range(len(market_data['returns'].columns))
        },
        group_constraints={
            'Equities': GroupConstraint(
                assets=[0, 1, 2, 3],
                bounds=(0.4, 0.7)  # 40-70% in equities
            ),
            'Bonds': GroupConstraint(
                assets=[4, 5, 6],
                bounds=(0.2, 0.5)  # 20-50% in bonds
            )
        },
        max_turnover=0.2  # 20% maximum turnover
    )
    
    # Run robust optimization
    result = optimizer.optimize(
        objective=ObjectiveFunction.GARLAPPI_ROBUST,
        constraints=constraints
    )
    
    # Calculate additional metrics
    robust_metrics = optimizer.calculate_robust_metrics(
        weights=result['weights']
    )
    
    return result, robust_metrics

def backtest_example():
    """
    Comprehensive backtesting example with robust optimization.
    
    Demonstrates:
    - Backtest setup
    - Rolling window optimization
    - Performance analysis
    - Visualization
    """
    # Prepare data
    market_data = generate_synthetic_multi_asset_data(
        n_years=10,  # Longer period for backtesting
        freq='M',
        seed=42
    )
    
    # Initialize backtest optimizer
    backtest = RobustBacktestOptimizer(
        returns=market_data['returns'],
        lookback_window=36,
        rebalance_frequency=3,
        transaction_cost=0.001
    )
    
    # Define constraints
    constraints = OptimizationConstraints(
        long_only=True,
        box_constraints={
            i: (0.0, 0.2)
            for i in range(len(market_data['returns'].columns))
        },
        max_turnover=0.15  # 15% turnover limit
    )
    
    # Run backtest
    results = backtest.run_backtest(
        objective=ObjectiveFunction.MEAN_VARIANCE,
        constraints=constraints
    )
    
    # Analyze results
    metrics = results['backtest_metrics']
    print("\nBacktest Results:")
    print(f"Total Return: {metrics['Total Return']:.2%}")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {metrics['Maximum Drawdown']:.2%}")
    
    # Create visualizations
    backtest.plot_backtest_results(results)
    
    return results

def efficient_frontier_example():
    """
    Example of efficient frontier analysis with robust optimization.
    
    Demonstrates:
    - Frontier computation
    - Risk-return analysis
    - Portfolio composition
    - Visualization
    """
    # Initialize data and optimizer
    market_data = generate_synthetic_multi_asset_data()
    frontier = RobustEfficientFrontier(
        returns=market_data['returns'],
        epsilon=0.1,
        optimization_method=OptimizationMethod.CVXPY
    )
    
    # Define constraints
    constraints = OptimizationConstraints(
        long_only=True,
        box_constraints={
            i: (0.0, 0.3)
            for i in range(len(market_data['returns'].columns))
        }
    )
    
    # Compute frontier
    results = frontier.compute_efficient_frontier(
        n_points=25,
        constraints=constraints,
        epsilon_range=(0.05, 0.15),
        risk_range=None  # Auto-compute from data
    )
    
    # Analyze and visualize
    frontier.plot_frontier(results)
    
    return results

def best_practices():
    """
    Best practices for using the portfolio optimization framework.
    """
    print("""
    Best Practices for Portfolio Optimization:
    
    1. Data Preparation:
       - Use sufficient historical data (minimum 2-3 years)
       - Clean and validate data thoroughly
       - Handle outliers appropriately
       - Consider forward-looking views
    
    2. Constraint Design:
       - Start with basic constraints
       - Add complexity gradually
       - Test constraint feasibility
       - Include reasonable bounds
    
    3. Optimization Parameters:
       - Use appropriate risk aversion levels
       - Consider parameter uncertainty
       - Test multiple scenarios
       - Validate results
    
    4. Risk Management:
       - Include transaction costs
       - Monitor turnover
       - Consider tracking error
       - Implement risk limits
    
    5. Backtesting:
       - Use appropriate window sizes
       - Consider transaction costs
       - Test multiple periods
       - Validate results
    
    6. Implementation:
       - Start simple and add complexity
       - Test edge cases
       - Monitor convergence
       - Document assumptions
    """)

def common_pitfalls():
    """
    Common pitfalls and how to avoid them.
    """
    print("""
    Common Pitfalls in Portfolio Optimization:
    
    1. Data Issues:
       - Insufficient history
       - Unhandled outliers
       - Look-ahead bias
       - Survivorship bias
    
    2. Constraint Issues:
       - Infeasible constraints
       - Over-constraining
       - Inconsistent constraints
       - Missing constraints
    
    3. Parameter Issues:
       - Unstable estimates
       - Extreme parameters
       - Ignored uncertainty
       - Poor scaling
    
    4. Implementation Issues:
       - Numerical instability
       - Convergence problems
       - Performance issues
       - Memory management
    
    5. Validation Issues:
       - Insufficient testing
       - Poor error handling
       - Missing edge cases
       - Incomplete documentation
    """)