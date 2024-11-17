import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
import cvxpy as cp
import scipy.stats
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum

@dataclass
class GroupConstraint:
    """
    Class for defining group-level constraints in portfolio optimization
    
    Attributes:
        assets: List of asset indices that belong to the group
        bounds: Tuple of (min_weight, max_weight) for group's total allocation
    """
    assets: List[int]  # List of asset indices in the group
    bounds: Tuple[float, float]  # (min_weight, max_weight) for group allocation

    def __post_init__(self):
        """Validate the constraint parameters"""
        if not isinstance(self.assets, list) or not all(isinstance(i, int) for i in self.assets):
            raise ValueError("assets must be a list of integers")
        
        if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
            raise ValueError("bounds must be a tuple of (min_weight, max_weight)")
            
        min_weight, max_weight = self.bounds
        if not (0 <= min_weight <= max_weight <= 1):
            raise ValueError("bounds must satisfy 0 <= min_weight <= max_weight <= 1")

    def validate_assets(self, n_assets: int):
        """Validate that asset indices are within bounds"""
        if not all(0 <= i < n_assets for i in self.assets):
            raise ValueError(f"Asset indices must be between 0 and {n_assets-1}")

@dataclass
class OptimizationConstraints:
    """
    Class for defining portfolio optimization constraints
    """
    group_constraints: Optional[Dict[str, GroupConstraint]] = None
    box_constraints: Optional[Dict[int, Tuple[float, float]]] = None
    long_only: bool = True
    max_turnover: Optional[float] = None
    target_risk: Optional[float] = None
    target_return: Optional[float] = None
    max_tracking_error: Optional[float] = None
    benchmark_weights: Optional[np.ndarray] = None

class OptimizationMethod(Enum):
    SCIPY = "scipy"
    CVXPY = "cvxpy"
    
class ObjectiveFunction(Enum):
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
    """Portfolio objective function factory"""
    
    @staticmethod
    def __calculate_estimation_error_covariance(returns: np.ndarray, method: str = 'bayes') -> np.ndarray:
        """
        Calculate the covariance matrix of estimation errors (Omega)
        """
        T, N = returns.shape
        
        if method == 'asymptotic':
            # Classical approach: Omega = Sigma/T
            sigma = np.cov(returns, rowvar=False)
            omega = sigma / T
            
        elif method == 'bayes':
            # Bayesian approach using sample variance
            mu = np.mean(returns, axis=0)
            sigma = np.cov(returns, rowvar=False)
            
            # Calculate sample variance of mean estimator
            sample_var = np.zeros((N, N))
            for t in range(T):
                dev = (returns[t] - mu).reshape(-1, 1)
                sample_var += dev @ dev.T
            
            # Bayesian posterior covariance
            omega = sample_var / (T * (T - 1))
            
        elif method == 'factor':
            # Factor-based approach using Principal Components
            k = min(3, N - 1)  # Number of factors
            
            # Perform PCA
            sigma = np.cov(returns, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(sigma)
            
            # Sort in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Separate systematic and idiosyncratic components
            systematic_var = eigenvectors[:, :k] @ np.diag(eigenvalues[:k]) @ eigenvectors[:, :k].T
            idiosyncratic_var = sigma - systematic_var
            
            # Estimation error covariance
            omega = systematic_var / T + np.diag(np.diag(idiosyncratic_var)) / T
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Ensure symmetry and positive definiteness
        omega = (omega + omega.T) / 2
        min_eigenval = np.min(np.linalg.eigvals(omega))
        if min_eigenval < 0:
            omega += (-min_eigenval + 1e-8) * np.eye(N)
        
        return omega
    
    @staticmethod
    def garlappi_robust(returns: np.ndarray, epsilon: float, alpha: np.ndarray,
                        omega_method: str = 'bayes', omega: Optional[np.ndarray] = None) -> callable:
        """
        Implements Garlappi et al. (2007) robust portfolio optimization with fixed inputs
        """
        # Ensure returns is 2-d numpy array
        if isinstance(returns, pd.DataFrame):
            returns = returns.values
        returns = np.atleast_2d(returns)
        
        # Calculate required inputs
        mu = np.mean(returns, axis=0)
        Sigma = np.cov(returns, rowvar=False)
        
        # Ensure alpha is proper array
        alpha = np.asarray(alpha).flatten()
        
        # Calculate Omega if not provided
        if omega is None:
            Omega = PortfolioObjective._PortfolioObjective__calculate_estimation_error_covariance(
                returns, 
                method=omega_method
            )
        else:
            Omega = omega
            
        def objective(w: np.ndarray) -> float:
            try:
                # Ensure w is 1-d array
                w = np.asarray(w).flatten()
                
                # Asset-specific risk penalties
                risk_penalties = np.diag(alpha) @ Sigma
                variance_penalty = 0.5 * w @ risk_penalties @ w
                
                # Handle numerical instability
                omega_w = Omega @ w
                omega_w_norm = np.sqrt(w @ Omega @ w)
                
                if omega_w_norm < 1e-8:
                    return 1e10  # Large penalty instead of infinity
                
                # Worst-case mean adjustment
                scaling = np.sqrt(epsilon * omega_w_norm)
                worst_case_mean = mu - scaling * omega_w / omega_w_norm
                
                # Complete objective
                robust_utility = w @ worst_case_mean - variance_penalty
                
                return -float(robust_utility)  # Ensure scalar output
                
            except Exception as e:
                print(f"Error in objective function: {str(e)}")
                return 1e10  # Return large penalty on error
                
        return objective
    
    @staticmethod
    def minimum_variance(Sigma: np.ndarray) -> callable:
        def objective(w: np.ndarray) -> float:
            return w.T @ Sigma @ w
        return objective
        
    @staticmethod
    def robust_mean_variance(mu: np.ndarray, Sigma: np.ndarray, epsilon: float, kappa: float) -> callable:
        def objective(w: np.ndarray) -> float:
            risk = np.sqrt(w.T @ Sigma @ w)
            return -mu.T @ w + kappa * risk - epsilon * risk
        return objective

    @staticmethod
    def maximum_sharpe(mu: np.ndarray, Sigma: np.ndarray, rf_rate: float = 0.0) -> callable:
        def objective(w: np.ndarray) -> float:
            risk = np.sqrt(w.T @ Sigma @ w)
            return -(mu.T @ w - rf_rate) / risk if risk > 0 else -np.inf
        return objective
        
    @staticmethod
    def minimum_tracking_error(Sigma: np.ndarray, benchmark: np.ndarray) -> callable:
        def objective(w: np.ndarray) -> float:
            diff = w - benchmark
            return diff.T @ Sigma @ diff
        return objective
        
    @staticmethod
    def maximum_quadratic_utility(mu: np.ndarray, Sigma: np.ndarray, risk_aversion: float = 1.0) -> callable:
        """
        Maximize quadratic utility function: U(w) = w'μ - (λ/2)w'Σw
        
        Args:
            mu: Expected returns
            Sigma: Covariance matrix
            risk_aversion: Risk aversion parameter (lambda)
        """
        def objective(w: np.ndarray) -> float:
            return -(mu.T @ w - (risk_aversion / 2) * w.T @ Sigma @ w)
        return objective

    @staticmethod
    def mean_variance(mu: np.ndarray, Sigma: np.ndarray, target_return: Optional[float] = None) -> callable:
        """
        Traditional Markowitz mean-variance optimization.
        If target_return is None, finds the optimal risk-return tradeoff.
        If target_return is specified, minimizes variance subject to return constraint.
        
        Args:
            mu: Expected returns
            Sigma: Covariance matrix
            target_return: Optional target return constraint
        """
        if target_return is None:
            def objective(w: np.ndarray) -> float:
                portfolio_return = mu.T @ w
                portfolio_variance = w.T @ Sigma @ w
                return portfolio_variance - portfolio_return
        else:
            def objective(w: np.ndarray) -> float:
                return w.T @ Sigma @ w
        return objective

    @staticmethod
    def maximum_diversification(Sigma: np.ndarray, asset_stdevs: Optional[np.ndarray] = None) -> callable:
        """
        Maximize the diversification ratio: (w'σ) / sqrt(w'Σw)
        where σ is the vector of asset standard deviations
        
        Args:
            Sigma: Covariance matrix
            asset_stdevs: Vector of asset standard deviations (if None, computed from Sigma)
        """
        if asset_stdevs is None:
            asset_stdevs = np.sqrt(np.diag(Sigma))
            
        def objective(w: np.ndarray) -> float:
            portfolio_risk = np.sqrt(w.T @ Sigma @ w)
            weighted_stdev_sum = w.T @ asset_stdevs
            return -weighted_stdev_sum / portfolio_risk if portfolio_risk > 0 else -np.inf
        return objective

    @staticmethod
    def minimum_cvar(returns: np.ndarray, alpha: float = 0.05, scenarios: Optional[np.ndarray] = None) -> callable:
        """
        Refined Conditional Value at Risk (CVaR) minimization with optional scenario analysis
        
        Args:
            returns: Historical returns matrix
            alpha: Confidence level (default: 0.05 for 95% CVaR)
            scenarios: Optional scenario returns for stress testing
        """
        n_samples = len(returns)
        cutoff_index = int(n_samples * alpha)
        
        if scenarios is not None:
            combined_returns = np.vstack([returns, scenarios])
        else:
            combined_returns = returns
            
        def objective(w: np.ndarray) -> float:
            portfolio_returns = combined_returns @ w
            sorted_returns = np.sort(portfolio_returns)
            # Calculate CVaR as the average of worst alpha% returns
            worst_returns = sorted_returns[:cutoff_index]
            cvar = -np.mean(worst_returns)
            return cvar
        return objective

    @staticmethod
    def mean_cvar(returns: np.ndarray, mu: np.ndarray, lambda_cvar: float = 0.5, 
                        alpha: float = 0.05, scenarios: Optional[np.ndarray] = None) -> callable:
        """
        Enhanced mean-CVaR optimization with scenario analysis and flexible risk-return tradeoff
        
        Args:
            returns: Historical returns matrix
            mu: Expected returns
            lambda_cvar: Risk-return tradeoff parameter (0 to 1)
            alpha: Confidence level for CVaR
            scenarios: Optional scenario returns for stress testing
        """
        cvar_obj = PortfolioObjective.minimum_cvar(returns, alpha, scenarios)
        
        def objective(w: np.ndarray) -> float:
            expected_return = mu.T @ w
            cvar_risk = cvar_obj(w)
            # Normalize the objectives to make lambda_cvar more interpretable
            return -(1 - lambda_cvar) * expected_return + lambda_cvar * cvar_risk
        return objective

    @staticmethod
    def risk_parity(Sigma: np.ndarray) -> callable:
        def risk_contribution(w: np.ndarray) -> np.ndarray:
            portfolio_risk = np.sqrt(w.T @ Sigma @ w)
            marginal_risk = Sigma @ w / portfolio_risk
            risk_contrib = w * marginal_risk
            return risk_contrib
        
        def objective(w: np.ndarray) -> float:
            rc = risk_contribution(w)
            rc_target = np.mean(rc)
            return np.sum((rc - rc_target) ** 2)
        return objective

    @staticmethod
    def equal_risk_contribution(Sigma: np.ndarray) -> callable:
        n = len(Sigma)
        target_risk = 1.0 / n
        
        def objective(w: np.ndarray) -> float:
            portfolio_risk = np.sqrt(w.T @ Sigma @ w)
            marginal_risk = Sigma @ w / portfolio_risk
            risk_contrib = w * marginal_risk / portfolio_risk
            return np.sum((risk_contrib - target_risk) ** 2)
        return objective

    @staticmethod
    def hierarchical_risk_parity(returns: np.ndarray, clusters: Optional[List[List[int]]] = None) -> callable:
        if clusters is None:
            clusters = HierarchicalRiskParity.get_clusters(returns)
            
        def objective(w: np.ndarray) -> float:
            cluster_variances = []
            for cluster in clusters:
                cluster_weight = np.sum(w[cluster])
                if cluster_weight > 0:
                    cluster_returns = returns[:, cluster] @ (w[cluster] / cluster_weight)
                    cluster_variances.append(np.var(cluster_returns))
            return np.std(cluster_variances) + np.mean(cluster_variances)
        return objective
    
class HierarchicalRiskParity:
    @staticmethod
    def get_clusters(returns: np.ndarray, n_clusters: int = 2) -> List[List[int]]:
        """Perform hierarchical clustering on assets"""
        from scipy.cluster.hierarchy import linkage, fcluster
        
        # Compute correlation-based distance matrix
        corr = np.corrcoef(returns.T)
        dist = np.sqrt(2 * (1 - corr))
        
        # Perform hierarchical clustering
        link = linkage(dist, method='ward')
        clusters = fcluster(link, n_clusters, criterion='maxclust')
        
        # Group assets by cluster
        cluster_groups = []
        for i in range(1, n_clusters + 1):
            cluster_groups.append(np.where(clusters == i)[0].tolist())
            
        return cluster_groups
        
    @staticmethod
    def get_quasi_diag(link: np.ndarray) -> List[int]:
        """Compute quasi-diagonal matrix for HRP"""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = sort_ix.append(df0)
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
            
        return sort_ix.tolist()
   
@dataclass
class GroupConstraint:
    """
    Class for defining group-level constraints in portfolio optimization
    
    Attributes:
        assets: List of asset indices that belong to the group
        bounds: Tuple of (min_weight, max_weight) for the group's total allocation
        
    Example:
        # Constraint: Sector 1 (assets 0,1,2) must be between 20% and 40% of portfolio
        sector1_constraint = GroupConstraint(
            assets=[0, 1, 2],
            bounds=(0.2, 0.4)
        )
    """
    assets: List[int]  # List of asset indices in the group
    bounds: Tuple[float, float]  # (min_weight, max_weight) for group allocation

    def __post_init__(self):
        """Validate the constraint parameters"""
        if not isinstance(self.assets, list) or not all(isinstance(i, int) for i in self.assets):
            raise ValueError("assets must be a list of integers")
        
        if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
            raise ValueError("bounds must be a tuple of (min_weight, max_weight)")
            
        min_weight, max_weight = self.bounds
        if not (0 <= min_weight <= max_weight <= 1):
            raise ValueError("bounds must satisfy 0 <= min_weight <= max_weight <= 1")

    def validate_assets(self, n_assets: int):
        """Validate that asset indices are within bounds"""
        if not all(0 <= i < n_assets for i in self.assets):
            raise ValueError(f"Asset indices must be between 0 and {n_assets-1}")
        
class PortfolioOptimizer:
    def __init__(
        self,
        returns: pd.DataFrame,
        optimization_method: OptimizationMethod = OptimizationMethod.SCIPY,
        half_life: int = 36,
        risk_free_rate: float = 0.0,
        transaction_cost: float = 0.001
    ):
        """
        Initialize portfolio optimizer
        
        Args:
            returns: DataFrame of asset returns (time x assets)
            optimization_method: SCIPY or CVXPY
            half_life: Half-life in months for covariance computation
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            transaction_cost: Transaction cost rate
        """
        self.returns = returns
        self.optimization_method = optimization_method
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        
        # Compute exponentially weighted covariance matrix
        self.covariance = self._compute_ewm_covariance(half_life)
        self.expected_returns = self._compute_expected_returns()
        
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
        Optimize portfolio based on selected objective and constraints
        
        Args:
            objective: Selected objective function
            constraints: Optimization constraints
            current_weights: Current portfolio weights
            **kwargs: Additional parameters for specific objective functions
        """
        if current_weights is None:
            current_weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
            
        if self.optimization_method == OptimizationMethod.SCIPY:
            return self._optimize_scipy(objective, constraints, current_weights, **kwargs)
        else:
            return self._optimize_cvxpy(objective, constraints, current_weights, **kwargs)
            
    def _compute_ewm_covariance(self, half_life: int) -> np.ndarray:
        """Compute exponentially weighted covariance matrix"""
        lambda_param = np.log(2) / half_life
        weights = np.exp(-lambda_param * np.arange(len(self.returns)))
        weights = weights / np.sum(weights)
        
        # Center returns
        centered_returns = self.returns - self.returns.mean()
        
        # Compute weighted covariance
        weighted_returns = centered_returns * np.sqrt(weights[:, np.newaxis])
        return weighted_returns.T @ weighted_returns
        
    def _compute_expected_returns(self) -> np.ndarray:
        """Compute expected returns using historical mean"""
        return self.returns.mean().values
    
    def _get_objective_function(
        self,
        objective: ObjectiveFunction,
        **kwargs
        ) -> Callable:
        """Get the appropriate objective function"""
        if objective == ObjectiveFunction.MINIMUM_VARIANCE:
            return self.objective_functions.minimum_variance(self.covariance)
        
        elif objective == ObjectiveFunction.GARLAPPI_ROBUST:
            epsilon = kwargs.get('epsilon', 0.1)  # Default uncertainty
            alpha = kwargs.get('alpha', 1.0)      # Default risk aversion
            omega_method = kwargs.get('omega_method', 'bayes')
            return self.objective_functions.garlappi_robust(
                returns=self.returns.values,
                epsilon=epsilon,
                alpha=alpha,
                omega_method=omega_method
            )
            
        elif objective == ObjectiveFunction.MEAN_VARIANCE:
            return self.objective_functions.mean_variance(
                self.expected_returns,
                self.covariance,
                kwargs.get('target_return')
            )
            
        elif objective == ObjectiveFunction.ROBUST_MEAN_VARIANCE:  # Add this block
            epsilon = kwargs.get('epsilon', 0.1)  # Default uncertainty
            kappa = kwargs.get('kappa', 1.0)      # Default risk aversion
            return self.objective_functions.robust_mean_variance(
                self.expected_returns,
                self.covariance,
                epsilon,
                kappa
            )
            
        elif objective == ObjectiveFunction.MAXIMUM_SHARPE:
            return self.objective_functions.maximum_sharpe(
                self.expected_returns,
                self.covariance,
                self.risk_free_rate
            )

        elif objective == ObjectiveFunction.MAXIMUM_QUADRATIC_UTILITY:
            risk_aversion = kwargs.get('risk_aversion', 1.0)
            return self.objective_functions.maximum_quadratic_utility(
                self.expected_returns,
                self.covariance,
                risk_aversion
            )
            
        elif objective == ObjectiveFunction.MINIMUM_TRACKING_ERROR:
            if 'benchmark_weights' not in kwargs:
                raise ValueError("benchmark_weights required for tracking error minimization")
            return self.objective_functions.minimum_tracking_error(
                self.covariance,
                kwargs['benchmark_weights']
            )
            
        elif objective == ObjectiveFunction.MAXIMUM_DIVERSIFICATION:
            return self.objective_functions.maximum_diversification(
                self.covariance,
                kwargs.get('asset_stdevs')
            )
            
        elif objective == ObjectiveFunction.MINIMUM_CVAR:
            if self.returns is None:
                raise ValueError("Historical returns required for CVaR optimization")
            return self.objective_functions.minimum_cvar(
                self.returns.values,
                kwargs.get('alpha', 0.05),
                kwargs.get('scenarios')
            )
            
        elif objective == ObjectiveFunction.MEAN_CVAR:
            if self.returns is None:
                raise ValueError("Historical returns required for Mean-CVaR optimization")
            return self.objective_functions.mean_cvar(
                self.returns.values,
                self.expected_returns,
                kwargs.get('lambda_cvar', 0.5),
                kwargs.get('alpha', 0.05),
                kwargs.get('scenarios')
            )
            
        elif objective == ObjectiveFunction.RISK_PARITY:
            return self.objective_functions.risk_parity(self.covariance)
            
        elif objective == ObjectiveFunction.EQUAL_RISK_CONTRIBUTION:
            return self.objective_functions.equal_risk_contribution(self.covariance)
            
        elif objective == ObjectiveFunction.HIERARCHICAL_RISK_PARITY:
            if self.returns is None:
                raise ValueError("Historical returns required for HRP optimization")
            return self.objective_functions.hierarchical_risk_parity(
                self.returns.values,
                kwargs.get('clusters')
            )
            
        else:
            raise ValueError(f"Unsupported objective function: {objective}")
        
    def _optimize_scipy(
        self,
        objective: ObjectiveFunction,
        constraints: OptimizationConstraints,
        current_weights: np.ndarray,
        **kwargs
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Optimize using scipy"""
        n_assets = len(self.returns.columns)
        
        # Get objective function
        obj_func = self._get_objective_function(objective, **kwargs)
        
        # Build constraints
        constraint_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Add constraints based on OptimizationConstraints
        if constraints.target_return is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: x @ self.expected_returns - constraints.target_return
            })
            
        if constraints.target_risk is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(x @ self.covariance @ x) - constraints.target_risk
            })
            
        if constraints.max_tracking_error is not None and constraints.benchmark_weights is not None:
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x: constraints.max_tracking_error - 
                               np.sqrt((x - constraints.benchmark_weights).T @ 
                                     self.covariance @ 
                                     (x - constraints.benchmark_weights))
            })
            
        if constraints.max_turnover is not None:
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x: constraints.max_turnover - np.sum(np.abs(x - current_weights))
            })
            
        # Build bounds
        if constraints.long_only:
            bounds = [(0, 1) for _ in range(n_assets)]
        else:
            bounds = [(-1, 1) for _ in range(n_assets)]
            
        if constraints.box_constraints:
            for idx, (min_w, max_w) in constraints.box_constraints.items():
                bounds[idx] = (min_w, max_w)
                
        # Optimize
        result = minimize(
            obj_func,
            current_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'ftol': 1e-8, 'maxiter': 1000}
        )
        
        if not result.success:
            raise ValueError("Optimization failed to converge")
            
        return self._calculate_metrics(result.x, current_weights, constraints)

    def _optimize_cvxpy(
        self,
        objective: ObjectiveFunction,
        constraints: OptimizationConstraints,
        current_weights: np.ndarray,
        **kwargs
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize portfolio using CVXPY
        
        Args:
            objective: Selected objective function
            constraints: Optimization constraints
            current_weights: Current portfolio weights
            **kwargs: Additional parameters for specific objectives
        """
        n_assets = len(self.returns.columns)
        w = cp.Variable(n_assets)
        
        # Build objective based on type
        if objective == ObjectiveFunction.MINIMUM_VARIANCE:
            obj = cp.Minimize(cp.quad_form(w, self.covariance))
            
        elif objective == ObjectiveFunction.MEAN_VARIANCE:
            if constraints.target_return is not None:
                obj = cp.Minimize(cp.quad_form(w, self.covariance))
            else:
                obj = cp.Minimize(cp.quad_form(w, self.covariance) - w @ self.expected_returns)
                
        elif objective == ObjectiveFunction.ROBUST_MEAN_VARIANCE:
            epsilon = kwargs.get('epsilon', 0.5)
            kappa = kwargs.get('kappa', 1.0)
            risk = cp.norm(self.covariance @ w)
            obj = cp.Minimize(-w @ self.expected_returns + kappa * risk - epsilon * risk)
            
        elif objective == ObjectiveFunction.MAXIMUM_SHARPE:
            risk = cp.norm(self.covariance @ w)
            ret = w @ self.expected_returns - self.risk_free_rate
            obj = cp.Maximize(ret / risk)
            
        elif objective == ObjectiveFunction.MAXIMUM_QUADRATIC_UTILITY:
            risk_aversion = kwargs.get('risk_aversion', 1.0)
            obj = cp.Maximize(w @ self.expected_returns - (risk_aversion/2) * cp.quad_form(w, self.covariance))
            
        elif objective == ObjectiveFunction.MINIMUM_TRACKING_ERROR:
            if 'benchmark_weights' not in kwargs:
                raise ValueError("benchmark_weights required for tracking error minimization")
            benchmark = kwargs['benchmark_weights']
            diff = w - benchmark
            obj = cp.Minimize(cp.quad_form(diff, self.covariance))
            
        elif objective == ObjectiveFunction.MAXIMUM_DIVERSIFICATION:
            asset_stdevs = kwargs.get('asset_stdevs', np.sqrt(np.diag(self.covariance)))
            portfolio_risk = cp.norm(self.covariance @ w)
            weighted_stdev_sum = w @ asset_stdevs
            obj = cp.Maximize(weighted_stdev_sum / portfolio_risk)
            
        elif objective == ObjectiveFunction.MINIMUM_CVAR:
            if self.returns is None:
                raise ValueError("Historical returns required for CVaR optimization")
            alpha = kwargs.get('alpha', 0.05)
            scenarios = kwargs.get('scenarios')
            
            # Use historical scenarios and additional stress scenarios if provided
            if scenarios is not None:
                scenario_returns = np.vstack([self.returns.values, scenarios])
            else:
                scenario_returns = self.returns.values
                
            n_scenarios = len(scenario_returns)
            aux_var = cp.Variable(1)  # VaR variable
            s = cp.Variable(n_scenarios)  # Auxiliary variables for CVaR
            
            # CVaR constraints
            cvar_constraints = [
                s >= 0,
                s >= -scenario_returns @ w - aux_var
            ]
            
            obj = cp.Minimize(aux_var + (1/(alpha * n_scenarios)) * cp.sum(s))
            
        elif objective == ObjectiveFunction.MEAN_CVAR:
            if self.returns is None:
                raise ValueError("Historical returns required for Mean-CVaR optimization")
            lambda_cvar = kwargs.get('lambda_cvar', 0.5)
            alpha = kwargs.get('alpha', 0.05)
            scenarios = kwargs.get('scenarios')
            
            # Combine historical and stress scenarios
            if scenarios is not None:
                scenario_returns = np.vstack([self.returns.values, scenarios])
            else:
                scenario_returns = self.returns.values
                
            n_scenarios = len(scenario_returns)
            aux_var = cp.Variable(1)  # VaR variable
            s = cp.Variable(n_scenarios)  # Auxiliary variables for CVaR
            
            # CVaR constraints
            cvar_constraints = [
                s >= 0,
                s >= -scenario_returns @ w - aux_var
            ]
            
            cvar_term = aux_var + (1/(alpha * n_scenarios)) * cp.sum(s)
            obj = cp.Minimize(-lambda_cvar * w @ self.expected_returns + (1-lambda_cvar) * cvar_term)
            
        elif objective == ObjectiveFunction.RISK_PARITY:
            # Approximate risk parity using convex optimization
            risk_target = 1.0 / n_assets
            portfolio_risk = cp.norm(self.covariance @ w)
            marginal_risk = self.covariance @ w / portfolio_risk
            risk_contrib = cp.multiply(w, marginal_risk)
            obj = cp.Minimize(cp.sum_squares(risk_contrib - risk_target))
            
        elif objective == ObjectiveFunction.EQUAL_RISK_CONTRIBUTION:
            # Similar to risk parity but with equal risk contribution
            target_risk = 1.0 / n_assets
            portfolio_risk = cp.norm(self.covariance @ w)
            marginal_risk = self.covariance @ w / portfolio_risk
            risk_contrib = cp.multiply(w, marginal_risk) / portfolio_risk
            obj = cp.Minimize(cp.sum_squares(risk_contrib - target_risk))
        
        elif objective == ObjectiveFunction.GARLAPPI_ROBUST:
            epsilon = kwargs.get('epsilon', 0.1)
            alpha = kwargs.get('alpha', 1.0)
            omega_method = kwargs.get('omega_method', 'bayes')
            
            # Calculate required inputs
            mu = self.expected_returns
            Sigma = self.covariance
            
            # Calculate Omega (estimation error covariance)
            Omega = self.objective_functions._PortfolioObjective__calculate_estimation_error_covariance(
                self.returns.values, 
                method=omega_method
            )
            # Regular mean-variance term
            variance_term = 0.5 * alpha * cp.quad_form(w, Sigma)
            
            # Worst-case mean adjustment
            omega_w_norm = cp.norm(Omega @ w)
            scaling = cp.sqrt(epsilon * omega_w_norm)
            worst_case_mean = mu - scaling * (Omega @ w) / omega_w_norm
            
            # Complete objective
            obj = cp.Maximize(w @ worst_case_mean - variance_term)
            
        else:
            raise ValueError(f"Objective function {objective} not implemented for CVXPY")
        
        # Build basic constraints
        constraint_list = [cp.sum(w) == 1]  # weights sum to 1
        
        # Add long-only constraint if specified
        if constraints.long_only:
            constraint_list.append(w >= 0)
        
        # Add target return constraint if specified
        if constraints.target_return is not None:
            constraint_list.append(w @ self.expected_returns == constraints.target_return)
        
        # Add target risk constraint if specified
        if constraints.target_risk is not None:
            constraint_list.append(cp.norm(self.covariance @ w) == constraints.target_risk)
        
        # Add tracking error constraint if specified
        if constraints.max_tracking_error is not None and constraints.benchmark_weights is not None:
            diff = w - constraints.benchmark_weights
            constraint_list.append(
                cp.norm(self.covariance @ diff) <= constraints.max_tracking_error
            )
        
        # Add turnover constraint if specified
        if constraints.max_turnover is not None:
            constraint_list.append(
                cp.norm(w - current_weights, 1) <= constraints.max_turnover
            )
        
        # Add box constraints if specified
        if constraints.box_constraints:
            for idx, (min_w, max_w) in constraints.box_constraints.items():
                constraint_list.extend([w[idx] >= min_w, w[idx] <= max_w])
        
        # Add group constraints if specified
        if constraints.group_constraints:
            for group in constraints.group_constraints.values():
                group_weight = cp.sum(w[group.assets])
                constraint_list.extend([
                    group_weight >= group.bounds[0],
                    group_weight <= group.bounds[1]
                ])
        
        # Add CVaR-specific constraints if needed
        if objective in [ObjectiveFunction.MINIMUM_CVAR, ObjectiveFunction.MEAN_CVAR]:
            constraint_list.extend(cvar_constraints)
        
        # Create and solve the problem
        try:
            prob = cp.Problem(obj, constraint_list)
            prob.solve()
            
            if prob.status != cp.OPTIMAL:
                raise ValueError(f"Optimization failed with status: {prob.status}")
                
            # Calculate metrics and return results
            return self._calculate_metrics(w.value, current_weights, constraints)
            
        except Exception as e:
            raise ValueError(f"CVXPY optimization failed: {str(e)}")
        
    def _calculate_metrics(
        self,
        weights: np.ndarray,
        current_weights: np.ndarray,
        constraints: OptimizationConstraints
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Calculate portfolio metrics"""
        portfolio_return = weights @ self.expected_returns
        portfolio_risk = np.sqrt(weights @ self.covariance @ weights)
        turnover = np.sum(np.abs(weights - current_weights))
        transaction_costs = self.transaction_cost * turnover
        
        metrics = {
            'weights': weights,
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_risk,
            'turnover': turnover,
            'transaction_costs': transaction_costs,
            'net_return': portfolio_return - transaction_costs
        }
        
        if constraints.benchmark_weights is not None:
            tracking_error = np.sqrt(
                (weights - constraints.benchmark_weights).T @ 
                self.covariance @ 
                (weights - constraints.benchmark_weights)
            )
            metrics['tracking_error'] = tracking_error
            
        return metrics

class RobustPortfolioOptimizer(PortfolioOptimizer):
    """Enhanced portfolio optimizer using Garlappi et al. (2007) robust optimization by default"""
    
    def __init__(
        self, 
        returns: pd.DataFrame, 
        epsilon: float = 0.1,
        alpha: Union[float, np.ndarray] = 1.0,  # Can be scalar or vector
        omega_method: str = 'bayes',
        optimization_method: OptimizationMethod = OptimizationMethod.SCIPY,
        half_life: int = 36,
        risk_free_rate: float = 0.0,
        transaction_cost: float = 0.001
    ):
        # Store original returns and parameters
        self.original_returns = returns.copy()
        self.epsilon = epsilon
        
        # Handle alpha as vector or scalar
        if isinstance(alpha, (int, float)):
            self.alpha = np.full(returns.shape[1], alpha)
        else:
            if len(alpha) != returns.shape[1]:
                raise ValueError(f"Alpha vector length ({len(alpha)}) must match number of assets ({returns.shape[1]})")
            self.alpha = np.array(alpha)
            
        self.omega_method = omega_method
        self.half_life = half_life
        
        # Initialize parent with processed returns
        super().__init__(
            returns=returns,
            optimization_method=optimization_method,
            half_life=half_life,
            risk_free_rate=risk_free_rate,
            transaction_cost=transaction_cost
        )
                
        # Calculate estimation error covariance (Omega)
        self.omega = self._calculate_estimation_error_covariance(returns.values,
            method=self.omega_method
        )
    
    def optimize(
        self, 
        objective: Optional[ObjectiveFunction] = None,
        constraints: Optional[OptimizationConstraints] = None,
        current_weights: Optional[np.ndarray] = None, 
        **kwargs
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Enhanced optimize method with Garlappi as default"""
        if current_weights is None:
            current_weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
            
        if constraints is None:
            constraints = OptimizationConstraints(long_only=True)
            
        # If no objective specified, use Garlappi robust
        if objective is None:
            objective = ObjectiveFunction.GARLAPPI_ROBUST
            kwargs.update({
                'epsilon': self.epsilon,
                'alpha': self.alpha,
                'omega': self.omega
            })
        
        # Force SCIPY for Garlappi optimization
        if objective == ObjectiveFunction.GARLAPPI_ROBUST:
            original_method = self.optimization_method
            self.optimization_method = OptimizationMethod.SCIPY
        
        try:
            # First attempt with specified parameters
            result = super().optimize(objective, constraints, current_weights, **kwargs)
            
            # Restore original optimization method
            if objective == ObjectiveFunction.GARLAPPI_ROBUST:
                self.optimization_method = original_method
                
            return result
            
        except ValueError as e:
            print(f"First optimization attempt failed: {e}")
            try:
                # Second attempt with relaxed constraints
                relaxed_constraints = self._relax_constraints(constraints)
                result = super().optimize(objective, relaxed_constraints, current_weights, **kwargs)
                
                # Restore original optimization method
                if objective == ObjectiveFunction.GARLAPPI_ROBUST:
                    self.optimization_method = original_method
                    
                return result
                
            except ValueError as e:
                print(f"Second optimization attempt failed: {e}")
                # Final attempt with Garlappi robust and minimal constraints
                print("Falling back to Garlappi robust optimization with minimal constraints...")
                return self._garlappi_robust_fallback(current_weights)
    
    def _relax_constraints(self, constraints: OptimizationConstraints) -> OptimizationConstraints:
        """Relax optimization constraints gradually"""
        relaxed = OptimizationConstraints(
            long_only=constraints.long_only,
            max_turnover=None if constraints.max_turnover else None,
            target_risk=None,
            target_return=None,
            max_tracking_error=None if constraints.max_tracking_error else None
        )
        
        # Relax box constraints if they exist
        if constraints.box_constraints:
            relaxed.box_constraints = {
                k: (max(0, v[0]-0.05), min(1, v[1]+0.05))
                for k, v in constraints.box_constraints.items()
            }
        
        # Relax group constraints if they exist
        if constraints.group_constraints:
            relaxed.group_constraints = {
                k: GroupConstraint(
                    assets=v.assets,
                    bounds=(max(0, v.bounds[0]-0.05), min(1, v.bounds[1]+0.05))
                )
                for k, v in constraints.group_constraints.items()
            }
        
        return relaxed
    
    def _garlappi_robust_fallback(self, current_weights: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """Fallback to Garlappi robust optimization with minimal constraints"""
        original_method = self.optimization_method
        self.optimization_method = OptimizationMethod.SCIPY
        
        constraints = OptimizationConstraints(
            long_only=True,
            box_constraints={i: (0, 0.3) for i in range(len(self.returns.columns))}
        )
        
        try:
            result = super().optimize(
                objective=ObjectiveFunction.GARLAPPI_ROBUST,
                constraints=constraints,
                current_weights=current_weights,
                epsilon=self.epsilon,
                alpha=self.alpha,
                omega=self.omega
            )
        finally:
            # Restore original optimization method
            self.optimization_method = original_method
            
        return result
    
    def _calculate_estimation_error_covariance(self, returns: np.ndarray, method: str = 'bayes') -> np.ndarray:
        """Calculate the covariance matrix of estimation errors (Omega)"""
        return PortfolioObjective._PortfolioObjective__calculate_estimation_error_covariance(
            returns=returns,
            method=method
        )
    
    def calculate_robust_metrics(self, weights: np.ndarray, alpha: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate additional robust performance metrics including Garlappi-specific metrics
        
        Args:
            weights: Portfolio weights
            alpha: Optional scaled alpha vector (if None, uses self.alpha)
        """
        # Use base alpha if no scaled alpha provided
        if alpha is None:
            alpha = self.alpha
            
        # Basic metrics with asset-specific risk aversion
        portfolio_return = weights @ self.expected_returns
        risk_matrix = np.diag(alpha) @ self.covariance if isinstance(alpha, np.ndarray) else alpha * self.covariance
        portfolio_risk = np.sqrt(weights.T @ risk_matrix @ weights)
        
        # Worst-case return using Garlappi framework
        omega_w = self.omega @ weights
        omega_w_norm = np.sqrt(weights.T @ self.omega @ weights)
        if omega_w_norm > 1e-8:
            scaling = np.sqrt(self.epsilon * omega_w_norm)
            worst_case_return = portfolio_return - scaling * omega_w_norm
        else:
            worst_case_return = portfolio_return
        
        # Diversification ratio with risk-adjusted standard deviations
        asset_stdevs = np.sqrt(np.diag(risk_matrix))
        div_ratio = (weights @ asset_stdevs) / portfolio_risk
        
        # Concentration metrics
        herfindahl = np.sum(weights ** 2)
        effective_n = 1 / herfindahl
        
        # Parameter uncertainty metrics
        estimation_uncertainty = np.sqrt(weights.T @ self.omega @ weights)
        
        # Asset-specific risk contributions
        marginal_risk = risk_matrix @ weights / portfolio_risk
        risk_contributions = weights * marginal_risk
        
        return {
            'worst_case_return': worst_case_return,
            'diversification_ratio': div_ratio,
            'herfindahl_index': herfindahl,
            'effective_n': effective_n,
            'estimation_uncertainty': estimation_uncertainty,
            'robust_sharpe': (worst_case_return - self.risk_free_rate) / portfolio_risk,
            'risk_contributions': risk_contributions
        }

class RobustEfficientFrontier(RobustPortfolioOptimizer):
    """Class for computing Garlappi robust efficient frontier with asset-specific risk aversion"""
    
    def __init__(self, returns: pd.DataFrame, **kwargs):
        super().__init__(returns, **kwargs)

    def compute_efficient_frontier(
            self,
            n_points: int = 15,
            epsilon_range: Optional[Tuple[float, float]] = None,
            risk_range: Optional[Tuple[float, float]] = None,
            alpha_scale_range: Optional[Tuple[float, float]] = None,
            constraints: Optional[OptimizationConstraints] = None
        ) -> Dict[str, np.ndarray]:
            """
            Compute the Garlappi robust efficient frontier with improved numerical stability
            """
            print("Computing Garlappi robust efficient frontier...")
            
            # Initialize base constraints if none provided
            if constraints is None:
                constraints = OptimizationConstraints(long_only=True)
            
            # Initialize containers
            frontier_results = {
                'returns': np.zeros(n_points),
                'risks': np.zeros(n_points),
                'sharpe_ratios': np.zeros(n_points),
                'weights': np.zeros((n_points, len(self.returns.columns))),
                'epsilons': np.zeros(n_points),
                'alpha_scales': np.zeros(n_points)
            }
            
            # Compute minimum variance portfolio for lower bound
            try:
                min_var_result = self.optimize(
                    objective=ObjectiveFunction.MINIMUM_VARIANCE,
                    constraints=constraints
                )
                min_risk = min_var_result['risk']
            except Exception as e:
                print(f"Failed to compute minimum variance portfolio: {e}")
                min_risk = np.sqrt(np.min(np.diag(self.covariance)))
            
            # Compute maximum return portfolio for upper bound
            try:
                max_return_weights = np.zeros(len(self.returns.columns))
                max_return_idx = np.argmax(self.expected_returns)
                max_return_weights[max_return_idx] = 1.0
                max_risk = np.sqrt(max_return_weights @ self.covariance @ max_return_weights)
            except Exception as e:
                print(f"Failed to compute maximum return portfolio: {e}")
                max_risk = np.sqrt(np.max(np.diag(self.covariance)))
            
            # Set risk range with buffer
            if risk_range is None:
                risk_range = (min_risk * 0.9, max_risk * 1.1)
            
            # Set epsilon range
            if epsilon_range is None:
                epsilon_range = (0.01, 0.5)  # More conservative range
            
            # Set alpha scale range
            if alpha_scale_range is None:
                alpha_scale_range = (0.8, 1.2)  # More conservative range
            
            valid_points = 0
            max_attempts = n_points * 2  # Allow for some failures
            attempt = 0
            
            while valid_points < n_points and attempt < max_attempts:
                try:
                    # Interpolate parameters
                    completion_ratio = valid_points / (n_points - 1) if n_points > 1 else 0
                    target_risk = risk_range[0] + (risk_range[1] - risk_range[0]) * completion_ratio
                    target_epsilon = epsilon_range[0] + (epsilon_range[1] - epsilon_range[0]) * completion_ratio
                    alpha_scale = alpha_scale_range[0] + (alpha_scale_range[1] - alpha_scale_range[0]) * completion_ratio
                    
                    # Scale alpha vector
                    scaled_alpha = self.alpha * alpha_scale
                    
                    # Add target risk to constraints with tolerance
                    risk_tolerance = target_risk * 0.1  # 10% tolerance
                    point_constraints = OptimizationConstraints(
                        long_only=True,
                        box_constraints={i: (0.0, 1.0) for i in range(len(self.returns.columns))},
                        target_risk=target_risk
                    )
                    
                    # Optimize portfolio with increased iterations and looser tolerance
                    result = self.optimize(
                        objective=ObjectiveFunction.GARLAPPI_ROBUST,
                        constraints=point_constraints,
                        epsilon=target_epsilon,
                        alpha=scaled_alpha,
                        current_weights=None,  # Start fresh each time
                        options={'maxiter': 1000, 'ftol': 1e-6}
                    )
                    
                    # Store results
                    frontier_results['returns'][valid_points] = result['return']
                    frontier_results['risks'][valid_points] = result['risk']
                    frontier_results['sharpe_ratios'][valid_points] = result['sharpe_ratio']
                    frontier_results['weights'][valid_points] = result['weights']
                    frontier_results['epsilons'][valid_points] = target_epsilon
                    frontier_results['alpha_scales'][valid_points] = alpha_scale
                    
                    valid_points += 1
                    print(f"Successfully computed point {valid_points}/{n_points}")
                    
                except Exception as e:
                    print(f"Failed attempt {attempt + 1}: {str(e)}")
                    attempt += 1
                    continue
                    
                attempt += 1
            
            if valid_points == 0:
                raise ValueError("Failed to compute any valid frontier points")
            
            # Trim results to valid points
            for key in frontier_results:
                frontier_results[key] = frontier_results[key][:valid_points]
            
            # Sort by risk
            sort_idx = np.argsort(frontier_results['risks'])
            for key in frontier_results:
                frontier_results[key] = frontier_results[key][sort_idx]
            
            return frontier_results
        
    def plot_frontier(self, frontier_results: Dict[str, np.ndarray]):
        """Create comprehensive Garlappi frontier visualization"""
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 2)
        
        # 1. Risk-Return Plot with Robust Sharpe Ratios
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_risk_return_robust(frontier_results, ax1)
        
        # 2. Uncertainty Impact
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_uncertainty_impact(frontier_results, ax2)
        
        # 3. Portfolio Composition
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_weights_evolution(frontier_results, ax3)
        
        # 4. Diversification and Uncertainty
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_diversification_uncertainty(frontier_results, ax4)
        
        plt.tight_layout()
        plt.show()
        
    def _plot_risk_return_robust(self, results: Dict[str, np.ndarray], ax: plt.Axes):
        """Plot risk-return frontier with robust Sharpe ratios"""
        sc = ax.scatter(
            results['risks'],
            results['returns'],
            c=results['robust_sharpe_ratios'],
            cmap='viridis',
            s=100
        )
        plt.colorbar(sc, ax=ax, label='Robust Sharpe Ratio')
        
        ax.set_xlabel('Risk (Volatility)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Garlappi Robust Efficient Frontier')
        ax.grid(True)
        
    def _plot_uncertainty_impact(self, results: Dict[str, np.ndarray], ax: plt.Axes):
        """Plot impact of uncertainty parameter"""
        ax.scatter(results['epsilons'], results['worst_case_returns'], 
                  c=results['risks'], cmap='viridis', label='Worst-Case Returns')
        plt.colorbar(ax.collections[0], ax=ax, label='Risk Level')
        
        ax.set_xlabel('Uncertainty Parameter (ε)')
        ax.set_ylabel('Worst-Case Return')
        ax.set_title('Impact of Uncertainty on Returns')
        ax.grid(True)
        
    def _plot_weights_evolution(self, results: Dict[str, np.ndarray], ax: plt.Axes):
        """Plot evolution of portfolio weights along the frontier"""
        weights = results['weights']
        risks = results['risks']
        
        for i in range(weights.shape[1]):
            ax.plot(risks, weights[:, i], label=f'Asset {i+1}')
            
        ax.set_xlabel('Risk (Volatility)')
        ax.set_ylabel('Weight')
        ax.set_title('Portfolio Composition Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax.grid(True)
        
    def _plot_diversification_uncertainty(self, results: Dict[str, np.ndarray], ax: plt.Axes):
        """Plot relationship between diversification and estimation uncertainty"""
        ax2 = ax.twinx()
        
        l1 = ax.plot(results['risks'], results['estimation_uncertainty'], 'b-',
                    label='Estimation Uncertainty')
        l2 = ax2.plot(results['risks'], results['effective_n'], 'r--',
                     label='Effective N')
        
        ax.set_xlabel('Risk (Volatility)')
        ax.set_ylabel('Estimation Uncertainty', color='b')
        ax2.set_ylabel('Effective N', color='r')
        
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)
        
        ax.set_title('Diversification and Parameter Uncertainty')
        ax.grid(True)

    def _get_risk_bounds(self) -> Tuple[float, float]:
        """Compute minimum and maximum risk bounds"""
        try:
            # Minimum risk portfolio
            min_var_result = self.optimize(
                objective=ObjectiveFunction.MINIMUM_VARIANCE,
                constraints=OptimizationConstraints(long_only=True)
            )
            min_risk = min_var_result['risk']
            
            # Maximum risk portfolio with reduced risk aversion
            reduced_alpha = self.alpha * 0.5
            max_risk_result = self.optimize(
                objective=ObjectiveFunction.GARLAPPI_ROBUST,
                constraints=OptimizationConstraints(
                    long_only=True,
                    box_constraints={i: (0, 1) for i in range(len(self.returns.columns))}
                ),
                epsilon=self.epsilon,
                alpha=reduced_alpha
            )
            
            # Use maximum individual asset risk as reference
            asset_stds = np.sqrt(np.diag(self.covariance))
            max_asset_risk = asset_stds.max()
            
            # Take larger of maximum portfolio risk and maximum asset risk
            max_risk = max(max_risk_result['risk'], max_asset_risk)
            
            # Add buffers
            max_risk *= 1.2
            min_risk *= 0.8
            
            return min_risk, max_risk
            
        except Exception as e:
            print(f"Error computing risk bounds: {str(e)}")
            asset_stds = np.sqrt(np.diag(self.covariance))
            return asset_stds.min() * 0.8, asset_stds.max() * 2

    def _create_frontier_constraints(
        self,
        base_constraints: OptimizationConstraints,
        target_risk: float
    ) -> OptimizationConstraints:
        """Create constraints for frontier point with target risk"""
        return OptimizationConstraints(
            group_constraints=base_constraints.group_constraints,
            box_constraints=base_constraints.box_constraints,
            long_only=base_constraints.long_only,
            max_turnover=base_constraints.max_turnover,
            target_risk=target_risk,
            max_tracking_error=base_constraints.max_tracking_error,
            benchmark_weights=base_constraints.benchmark_weights
        )

    def calculate_frontier_risk_contributions(self, frontier_results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Calculate and analyze risk contributions across the frontier"""
        n_points, n_assets = frontier_results['weights'].shape
        contributions = frontier_results['risk_contributions']
        
        return pd.DataFrame(
            contributions,
            columns=[f'Asset_{i}' for i in range(n_assets)],
            index=[f'Point_{i}' for i in range(n_points)]
        )

class RobustBacktestOptimizer(RobustPortfolioOptimizer):
    """Enhanced class for performing vectorized backtesting of portfolio optimization strategies"""
    
    def __init__(
        self,
        returns: pd.DataFrame,
        lookback_window: int = 36,
        rebalance_frequency: int = 3,
        estimation_method: str = 'robust',
        transaction_cost: float = 0.001,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
        epsilon: float = 0.1,
        **kwargs
    ):
        """
        Initialize backtester with simplified, robust design
        
        Args:
            returns: DataFrame of asset returns (time x assets)
            lookback_window: Number of periods for parameter estimation
            rebalance_frequency: Number of periods between rebalances
            estimation_method: 'robust' or 'standard'
            transaction_cost: Transaction cost rate
            benchmark_returns: Optional benchmark return series
            risk_free_rate: Risk-free rate
            epsilon: Uncertainty parameter for robust optimization
            **kwargs: Additional parameters for parent class
        """
        # Initialize parent with full dataset
        super().__init__(
            returns=returns,
            epsilon=epsilon,
            risk_free_rate=risk_free_rate,
            transaction_cost=transaction_cost,
            **kwargs
        )
        
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        self.estimation_method = estimation_method
        self.benchmark_returns = benchmark_returns
        
        # Validation
        if lookback_window >= len(returns):
            raise ValueError("Lookback window must be shorter than returns history")
            
    def _estimate_parameters(self, historical_returns: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Estimate parameters using selected method
        
        Args:
            historical_returns: Historical return data
        """
        if self.estimation_method == 'robust':
            # Use median for returns and Ledoit-Wolf for covariance
            mu = historical_returns.median()
            
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            sigma = lw.fit(historical_returns).covariance_
            
        else:
            # Standard estimation
            mu = historical_returns.mean()
            sigma = historical_returns.cov()
            
        return {
            'expected_returns': mu.values,
            'covariance': sigma.values if isinstance(sigma, pd.DataFrame) else sigma
        }
        
    def run_backtest(
            self,
            objective: ObjectiveFunction,
            constraints: OptimizationConstraints,
            initial_weights: Optional[np.ndarray] = None,
            **kwargs
        ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
            """Run backtest with corrected array handling"""
            # Convert returns index to timezone-naive datetime
            returns = self.returns.copy()
            returns.index = returns.index.tz_localize(None)
            
            dates = returns.index
            n_assets = len(returns.columns)
            
            # Initialize weights properly as 1-d array
            if initial_weights is None:
                initial_weights = np.ones(n_assets) / n_assets
            else:
                initial_weights = np.asarray(initial_weights).flatten()
                
            # Initialize results containers
            portfolio_weights = pd.DataFrame(
                0.0, 
                index=dates, 
                columns=returns.columns,
                dtype=np.float64
            )
            portfolio_returns = pd.Series(0.0, index=dates, dtype=np.float64)
            realized_costs = pd.Series(0.0, index=dates, dtype=np.float64)
            optimization_metrics = pd.DataFrame(
                0.0,
                index=dates,
                columns=['expected_return', 'expected_risk', 'sharpe_ratio'],
                dtype=np.float64
            )
            
            current_weights = initial_weights.copy()
            
            print("Running backtest...")
            try:
                for t in tqdm(range(self.lookback_window, len(dates))):
                    current_date = dates[t]
                    
                    # Determine if rebalancing is needed
                    should_rebalance = (t - self.lookback_window) % self.rebalance_frequency == 0
                    
                    if should_rebalance:
                        try:
                            # Get historical data for estimation
                            historical_returns = returns.iloc[t-self.lookback_window:t]
                            
                            # Ensure returns are properly formatted as 2-d numpy array
                            if isinstance(historical_returns, pd.DataFrame):
                                historical_returns_array = historical_returns.values
                            else:
                                historical_returns_array = np.atleast_2d(historical_returns)
                                
                            if len(historical_returns_array.shape) == 1:
                                historical_returns_array = historical_returns_array.reshape(-1, 1)
                            
                            # Create temporary optimizer with proper array shapes
                            temp_optimizer = RobustPortfolioOptimizer(
                                returns=pd.DataFrame(
                                    historical_returns_array,
                                    index=historical_returns.index,
                                    columns=returns.columns
                                ),
                                epsilon=self.epsilon,
                                risk_free_rate=self.risk_free_rate,
                                transaction_cost=self.transaction_cost
                            )
                            
                            # Optimize portfolio with error handling
                            try:
                                result = temp_optimizer.optimize(
                                    objective=objective,
                                    constraints=constraints,
                                    current_weights=current_weights.flatten(),  # Ensure 1-d
                                    **kwargs
                                )
                                
                                # Ensure weights are 1-d
                                new_weights = np.asarray(result['weights']).flatten()
                                
                                # Record metrics
                                optimization_metrics.loc[current_date] = {
                                    'expected_return': float(result['return']),  # Ensure scalar
                                    'expected_risk': float(result['risk']),     # Ensure scalar
                                    'sharpe_ratio': float(result['sharpe_ratio']) # Ensure scalar
                                }
                                
                                # Calculate and record costs
                                costs = float(self.transaction_cost * np.sum(np.abs(new_weights - current_weights)))
                                realized_costs.loc[current_date] = costs
                                
                                # Update weights
                                current_weights = new_weights
                                
                            except Exception as e:
                                print(f"Optimization failed at {current_date}: {str(e)}")
                                # Keep current weights if optimization fails
                        
                        except Exception as e:
                            print(f"Historical data processing failed at {current_date}: {str(e)}")
                            continue
                    
                    # Record weights
                    portfolio_weights.loc[current_date] = pd.Series(
                        current_weights, 
                        index=returns.columns
                    )
                    
                    # Calculate realized return (ensure proper broadcasting)
                    period_return = returns.loc[current_date].values.flatten()
                    portfolio_returns.loc[current_date] = float(
                        np.dot(period_return, current_weights) - 
                        realized_costs.loc[current_date]
                    )
                    
            except KeyboardInterrupt:
                print("\nBacktest interrupted by user")
                
            # Clean up results
            portfolio_returns = portfolio_returns.fillna(0)
            portfolio_weights = portfolio_weights.fillna(method='ffill')
            optimization_metrics = optimization_metrics.fillna(method='ffill')
            realized_costs = realized_costs.fillna(0)
            
            # Calculate final metrics
            metrics = self._calculate_backtest_metrics(
                portfolio_returns=portfolio_returns,
                portfolio_weights=portfolio_weights,
                realized_costs=realized_costs
            )
        
            return {
                'returns': portfolio_returns.to_frame('returns'),
                'weights': portfolio_weights,
                'metrics_history': optimization_metrics,
                'realized_costs': realized_costs.to_frame('costs'),
                'backtest_metrics': pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])
            }

    def save_backtest_results(self, results: Dict[str, Union[pd.Series, pd.DataFrame]], 
                            filename: str):
        """Save backtest results to file with timezone handling"""
        # Convert all datetime indices to timezone-naive
        results_to_save = {}
        for key, data in results.items():
            if isinstance(data, (pd.DataFrame, pd.Series)):
                if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
                    data = data.copy()
                    data.index = data.index.tz_localize(None)
            results_to_save[key] = data
        
        with pd.ExcelWriter(filename) as writer:
            for sheet_name, data in results_to_save.items():
                data.to_excel(writer, sheet_name=sheet_name)
        
    def _calculate_backtest_metrics(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights: pd.DataFrame,
        realized_costs: pd.Series
    ) -> Dict[str, float]:
        """Calculate key backtest metrics with proper array handling"""
        
        # Ensure inputs are properly formatted
        portfolio_returns = portfolio_returns.fillna(0).astype(float)
        portfolio_weights = portfolio_weights.fillna(0).astype(float)
        realized_costs = realized_costs.fillna(0).astype(float)
        
        # Calculate returns metrics (ensure scalar outputs)
        total_return = float((1 + portfolio_returns).prod() - 1)
        ann_return = float((1 + total_return) ** (12 / len(portfolio_returns)) - 1)
        volatility = float(portfolio_returns.std() * np.sqrt(12))
        sharpe = float((ann_return - self.risk_free_rate) / volatility if volatility > 0 else 0)
        
        # Calculate drawdowns
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = float(drawdowns.min())
        
        # Calculate cost metrics
        total_costs = float(realized_costs.sum())
        turnover = float(portfolio_weights.diff().abs().sum(axis=1).mean())
        
        return {
            'Total Return': total_return,
            'Annualized Return': ann_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Maximum Drawdown': max_drawdown,
            'Average Turnover': turnover,
            'Total Costs': total_costs
        }

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