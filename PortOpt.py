import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
import cvxpy as cp
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
            epsilon = kwargs.get('epsilon', self.uncertainty)
            kappa = kwargs.get('kappa', self.risk_aversion)
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

def preprocess_covariance(returns: pd.DataFrame, half_life: int = 36) -> np.ndarray:
    """
    Compute and regularize the covariance matrix
    
    Args:
        returns: DataFrame of asset returns
        half_life: Half-life for exponential weighting
    """
    # 1. Exponentially weighted covariance
    lambda_param = np.log(2) / half_life
    weights = np.exp(-lambda_param * np.arange(len(returns)))
    weights = weights / np.sum(weights)
    
    # Center returns
    centered_returns = returns - returns.mean()
    
    # Compute weighted covariance
    weighted_returns = centered_returns * np.sqrt(weights[:, np.newaxis])
    cov = weighted_returns.T @ weighted_returns
    
    # 2. Regularization using shrinkage
    n = len(returns)
    sample_cov = cov.values
    target = np.diag(np.diag(sample_cov))  # Diagonal target
    
    # Compute optimal shrinkage intensity
    phi = np.sum((sample_cov - target) ** 2)
    gamma = np.linalg.norm(sample_cov - target, 'fro')
    kappa = phi / gamma
    delta = min(1, kappa / n)
    
    # Apply shrinkage
    shrunk_cov = (1 - delta) * sample_cov + delta * target
    
    # 3. Ensure positive definiteness
    eigenvals = np.linalg.eigvals(shrunk_cov)
    if min(eigenvals) < 1e-8:
        min_eig = min(eigenvals)
        shrunk_cov += (1e-8 - min_eig) * np.eye(len(shrunk_cov))
    
    return shrunk_cov

class RobustPortfolioOptimizer(PortfolioOptimizer):
    """Enhanced portfolio optimizer with robust optimization methods"""
    
    def __init__(
        self, 
        returns: pd.DataFrame, 
        uncertainty: float = 0.1, 
        risk_aversion: float = 1.0,
        optimization_method: OptimizationMethod = OptimizationMethod.SCIPY,
        half_life: int = 36,
        risk_free_rate: float = 0.0,
        transaction_cost: float = 0.001
    ):
        # Store original returns and parameters
        self.original_returns = returns.copy()
        self.uncertainty = uncertainty
        self.risk_aversion = risk_aversion
        self.half_life = half_life
        
        # Preprocess returns
        self.processed_returns = self._preprocess_returns(returns)
        
        # Initialize parent with processed returns
        super().__init__(
            returns=self.processed_returns,
            optimization_method=optimization_method,
            half_life=half_life,
            risk_free_rate=risk_free_rate,
            transaction_cost=transaction_cost
        )
        
        # Override covariance with robust estimation
        self.covariance = self._preprocess_covariance()
        
    def _preprocess_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive returns preprocessing
        
        1. Outlier detection and handling
        2. Missing value treatment
        3. Volatility adjustment
        """
        processed = returns.copy()
        
        # Handle missing values
        processed = processed.fillna(method='ffill').fillna(method='bfill')
        
        # Detect and handle outliers using Winsorization
        for column in processed.columns:
            # Calculate robust statistics
            median = processed[column].median()
            mad = np.median(np.abs(processed[column] - median))
            lower_bound = median - 3 * mad
            upper_bound = median + 3 * mad
            
            # Winsorize outliers
            processed[column] = processed[column].clip(lower=lower_bound, upper=upper_bound)
        
        # Volatility adjustment using exponential weighting
        vol_adj_returns = processed.copy()
        lambda_param = np.log(2) / self.half_life
        weights = np.exp(-lambda_param * np.arange(len(processed)))[::-1]
        weights = weights / np.sum(weights)
        
        for column in processed.columns:
            rolling_vol = np.sqrt(
                np.sum(weights * (processed[column].values ** 2))
            )
            vol_adj_returns[column] = processed[column] / rolling_vol
            
        return vol_adj_returns
    
    def _preprocess_covariance(self) -> np.ndarray:
        """
        Robust covariance matrix estimation
        
        1. Exponential weighting
        2. Shrinkage estimation
        3. Conditioning improvement
        4. Positive definiteness enforcement
        """
        # 1. Exponentially weighted covariance
        lambda_param = np.log(2) / self.half_life
        weights = np.exp(-lambda_param * np.arange(len(self.processed_returns)))[::-1]
        weights = weights / np.sum(weights)
        
        # Center returns
        centered_returns = self.processed_returns - self.processed_returns.mean()
        
        # Compute weighted covariance
        weighted_returns = centered_returns * np.sqrt(weights[:, np.newaxis])
        sample_cov = weighted_returns.T @ weighted_returns
        
        # 2. Shrinkage estimation
        n_assets = len(self.processed_returns.columns)
        sample_cov_values = sample_cov.values
        
        # Compute shrinkage target (diagonal matrix)
        target = np.diag(np.diag(sample_cov_values))
        
        # Compute optimal shrinkage intensity
        phi = np.sum((sample_cov_values - target) ** 2)
        gamma = np.linalg.norm(sample_cov_values - target, 'fro')
        kappa = phi / gamma
        delta = min(1, kappa / n_assets)
        
        # Apply shrinkage
        shrunk_cov = (1 - delta) * sample_cov_values + delta * target
        
        # 3. Improve conditioning
        eigenvals, eigenvecs = np.linalg.eigh(shrunk_cov)
        
        # Set minimum eigenvalue threshold
        min_eigenval = max(1e-8, eigenvals.max() * 1e-6)
        eigenvals = np.maximum(eigenvals, min_eigenval)
        
        # Reconstruct covariance matrix
        cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # 4. Ensure symmetry and positive definiteness
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        return cov_matrix
    
    def optimize(
        self, 
        objective: ObjectiveFunction, 
        constraints: OptimizationConstraints,
        current_weights: Optional[np.ndarray] = None, 
        **kwargs
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Enhanced optimize method with multiple attempts and robust fallback"""
        if current_weights is None:
            current_weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
        
        try:
            # First attempt with original parameters
            return super().optimize(objective, constraints, current_weights, **kwargs)
        except ValueError as e:
            print(f"First optimization attempt failed: {e}")
            try:
                # Second attempt with relaxed constraints
                relaxed_constraints = self._relax_constraints(constraints)
                return super().optimize(objective, relaxed_constraints, current_weights, **kwargs)
            except ValueError as e:
                print(f"Second optimization attempt failed: {e}")
                # Final attempt with robust mean-variance
                print("Falling back to robust mean-variance optimization...")
                return self._robust_mean_variance_fallback(current_weights)
    
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
    
    def _robust_mean_variance_fallback(self, current_weights: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """Fallback to robust mean-variance optimization with minimal constraints"""
        constraints = OptimizationConstraints(
            long_only=True,
            box_constraints={i: (0, 0.3) for i in range(len(self.returns.columns))}
        )
        
        result = super().optimize(
            objective=ObjectiveFunction.ROBUST_MEAN_VARIANCE,
            constraints=constraints,
            current_weights=current_weights,
            epsilon=self.uncertainty,
            kappa=self.risk_aversion
        )
        
        return result
    
    def calculate_robust_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate additional robust performance metrics"""
        # Basic metrics
        portfolio_return = weights @ self.expected_returns
        portfolio_risk = np.sqrt(weights @ self.covariance @ weights)
        
        # Worst-case return
        worst_case_return = portfolio_return - self.uncertainty * portfolio_risk
        
        # Diversification ratio
        asset_stdevs = np.sqrt(np.diag(self.covariance))
        div_ratio = (weights @ asset_stdevs) / portfolio_risk
        
        # Concentration metrics
        herfindahl = np.sum(weights ** 2)
        effective_n = 1 / herfindahl
        
        return {
            'worst_case_return': worst_case_return,
            'diversification_ratio': div_ratio,
            'herfindahl_index': herfindahl,
            'effective_n': effective_n
        }
        
class RobustEfficientFrontier(RobustPortfolioOptimizer):
    """Class for computing robust efficient frontier using Robust Mean-Variance optimization"""
    
    def __init__(self, returns: pd.DataFrame, **kwargs):
        super().__init__(returns, **kwargs)
        
    def compute_efficient_frontier(
        self,
        n_points: int = 15,
        risk_range: Optional[Tuple[float, float]] = None,
        return_range: Optional[Tuple[float, float]] = None,
        constraints: Optional[OptimizationConstraints] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute the robust efficient frontier
        
        Args:
            n_points: Number of points on the frontier
            risk_range: Optional (min_risk, max_risk) tuple
            return_range: Optional (min_return, max_return) tuple
            constraints: Basic constraints for all portfolios
            
        Returns:
            Dictionary containing frontier information
        """
        print("Computing robust efficient frontier...")
        
        # Initialize base constraints if none provided
        if constraints is None:
            constraints = OptimizationConstraints(long_only=True)
        
        # Get risk range if not provided
        if risk_range is None:
            risk_range = self._get_risk_bounds()
            
        # Get return range if not provided
        if return_range is None:
            return_range = self._get_return_bounds()
            
        # Initialize result containers
        frontier_results = {
            'returns': np.zeros(n_points),
            'risks': np.zeros(n_points),
            'sharpe_ratios': np.zeros(n_points),
            'worst_case_returns': np.zeros(n_points),
            'weights': np.zeros((n_points, len(self.returns.columns))),
            'diversification_ratios': np.zeros(n_points),
            'effective_n': np.zeros(n_points)
        }
        
        # Compute frontier points
        for i in tqdm(range(n_points)):
            try:
                # Interpolate target return
                target_return = return_range[0] + (return_range[1] - return_range[0]) * i / (n_points - 1)
                
                # Create constraints with target return
                point_constraints = self._create_frontier_constraints(constraints, target_return)
                
                # Optimize portfolio
                result = self.optimize(
                    objective=ObjectiveFunction.ROBUST_MEAN_VARIANCE,
                    constraints=point_constraints,
                    epsilon=self.uncertainty,
                    kappa=self.risk_aversion
                )
                
                # Store results
                frontier_results['returns'][i] = result['return']
                frontier_results['risks'][i] = result['risk']
                frontier_results['sharpe_ratios'][i] = result['sharpe_ratio']
                frontier_results['weights'][i] = result['weights']
                
                # Calculate and store robust metrics
                robust_metrics = self.calculate_robust_metrics(result['weights'])
                frontier_results['worst_case_returns'][i] = robust_metrics['worst_case_return']
                frontier_results['diversification_ratios'][i] = robust_metrics['diversification_ratio']
                frontier_results['effective_n'][i] = robust_metrics['effective_n']
                
            except Exception as e:
                print(f"Failed to compute frontier point {i}: {str(e)}")
                continue
        
        # Clean up any failed points
        mask = frontier_results['risks'] > 0
        for key in frontier_results:
            frontier_results[key] = frontier_results[key][mask]
            
        # Sort by risk
        sort_idx = np.argsort(frontier_results['risks'])
        for key in frontier_results:
            frontier_results[key] = frontier_results[key][sort_idx]
            
        return frontier_results
    
    def _get_risk_bounds(self) -> Tuple[float, float]:
        """Compute minimum and maximum risk bounds"""
        try:
            # Minimum risk portfolio
            min_var_result = self.optimize(
                objective=ObjectiveFunction.MINIMUM_VARIANCE,
                constraints=OptimizationConstraints(long_only=True)
            )
            min_risk = min_var_result['risk']
            
            # Maximum return portfolio for max risk
            max_return_result = self.optimize(
                objective=ObjectiveFunction.ROBUST_MEAN_VARIANCE,
                constraints=OptimizationConstraints(
                    long_only=True,
                    box_constraints={i: (0, 1) for i in range(len(self.returns.columns))}
                ),
                epsilon=0,  # No uncertainty penalty for maximum risk
                kappa=0    # No risk aversion for maximum risk
            )
            max_risk = max_return_result['risk'] * 1.2  # Add 20% buffer
            
            return min_risk, max_risk
            
        except Exception as e:
            print(f"Error computing risk bounds: {str(e)}")
            # Fallback to standard deviation range
            asset_stds = np.sqrt(np.diag(self.covariance))
            return asset_stds.min(), asset_stds.max() * 2
    
    def _get_return_bounds(self) -> Tuple[float, float]:
        """Compute minimum and maximum return bounds"""
        try:
            # Get return range from expected returns
            min_return = self.expected_returns.min()
            max_return = self.expected_returns.max()
            
            # Add buffer
            return_range = max_return - min_return
            min_return -= return_range * 0.1
            max_return += return_range * 0.1
            
            return min_return, max_return
            
        except Exception as e:
            print(f"Error computing return bounds: {str(e)}")
            return 0, self.expected_returns.max() * 1.2
    
    def _create_frontier_constraints(
        self,
        base_constraints: OptimizationConstraints,
        target_return: float
    ) -> OptimizationConstraints:
        """Create constraints for frontier point"""
        return OptimizationConstraints(
            group_constraints=base_constraints.group_constraints,
            box_constraints=base_constraints.box_constraints,
            long_only=base_constraints.long_only,
            max_turnover=base_constraints.max_turnover,
            target_return=target_return,
            max_tracking_error=base_constraints.max_tracking_error,
            benchmark_weights=base_constraints.benchmark_weights
        )
    
    def plot_frontier(self, frontier_results: Dict[str, np.ndarray]):
        """Create comprehensive frontier visualization"""
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 2)
        
        # 1. Risk-Return Plot
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_risk_return(frontier_results, ax1)
        
        # 2. Worst-Case Returns
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_worst_case_returns(frontier_results, ax2)
        
        # 3. Portfolio Composition
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_weights_evolution(frontier_results, ax3)
        
        # 4. Diversification Metrics
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_diversification_metrics(frontier_results, ax4)
        
        plt.tight_layout()
        plt.show()
        
    def _plot_risk_return(self, results: Dict[str, np.ndarray], ax: plt.Axes):
        """Plot risk-return frontier with Sharpe ratio information"""
        sc = ax.scatter(
            results['risks'],
            results['returns'],
            c=results['sharpe_ratios'],
            cmap='viridis',
            s=100
        )
        plt.colorbar(sc, ax=ax, label='Sharpe Ratio')
        
        ax.set_xlabel('Risk (Volatility)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Robust Efficient Frontier')
        ax.grid(True)
        
    def _plot_worst_case_returns(self, results: Dict[str, np.ndarray], ax: plt.Axes):
        """Plot worst-case returns against expected returns"""
        ax.plot(results['risks'], results['returns'], 'b-', label='Expected Returns')
        ax.plot(results['risks'], results['worst_case_returns'], 'r--', label='Worst-Case Returns')
        ax.fill_between(results['risks'], results['worst_case_returns'], results['returns'], alpha=0.2)
        
        ax.set_xlabel('Risk (Volatility)')
        ax.set_ylabel('Return')
        ax.set_title('Expected vs Worst-Case Returns')
        ax.legend()
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
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
    def _plot_diversification_metrics(self, results: Dict[str, np.ndarray], ax: plt.Axes):
        """Plot diversification metrics along the frontier"""
        ax2 = ax.twinx()
        
        l1 = ax.plot(results['risks'], results['diversification_ratios'], 'b-',
                    label='Diversification Ratio')
        l2 = ax2.plot(results['risks'], results['effective_n'], 'r--',
                     label='Effective N')
        
        ax.set_xlabel('Risk (Volatility)')
        ax.set_ylabel('Diversification Ratio', color='b')
        ax2.set_ylabel('Effective N', color='r')
        
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)
        
        ax.set_title('Diversification Metrics')
        ax.grid(True)

class RobustBacktestOptimizer(RobustPortfolioOptimizer):
    """Class for performing vectorized backtesting of portfolio optimization strategies"""
    
    def __init__(
        self,
        returns: pd.DataFrame,
        lookback_window: int = 36,  # 3 years of momthly data
        rebalance_frequency: int = 3,  # Quarterly rebalancing
        **kwargs
    ):
        super().__init__(returns=returns, **kwargs)
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        
    def run_backtest(
        self,
        objective: ObjectiveFunction,
        constraints: OptimizationConstraints,
        **kwargs
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Run vectorized backtest of portfolio strategy
        
        Args:
            objective: Portfolio objective function
            constraints: Portfolio constraints
            **kwargs: Additional parameters for optimization
            
        Returns:
            Dictionary containing backtest results
        """
        returns = self.returns
        dates = returns.index
        n_assets = len(returns.columns)
        
        # Initialize result containers
        portfolio_weights = pd.DataFrame(index=dates, columns=returns.columns)
        portfolio_returns = pd.Series(index=dates, dtype=float)
        metrics_history = pd.DataFrame(index=dates)
        
        # Initialize with equal weights
        current_weights = np.ones(n_assets) / n_assets
        portfolio_weights.iloc[0] = current_weights
        
        print("Running backtest...")
        for t in tqdm(range(self.lookback_window, len(returns)), desc="Backtesting"):
            # Check if rebalancing is needed
            if (t - self.lookback_window) % self.rebalance_frequency == 0:
                # Get historical data for optimization
                historical_returns = returns.iloc[t-self.lookback_window:t]
                
                try:
                    # Create temporary optimizer for this period
                    period_optimizer = RobustPortfolioOptimizer(
                        returns=historical_returns,
                        uncertainty=self.uncertainty,
                        risk_aversion=self.risk_aversion,
                        optimization_method=self.optimization_method,
                        half_life=self.half_life,
                        risk_free_rate=self.risk_free_rate,
                        transaction_cost=self.transaction_cost
                    )
                    
                    # Optimize portfolio
                    result = period_optimizer.optimize(
                        objective=objective,
                        constraints=constraints,
                        current_weights=current_weights,
                        **kwargs
                    )
                    
                    # Update weights
                    current_weights = result['weights']
                    
                    # Store metrics
                    for key, value in result.items():
                        if key != 'weights':
                            metrics_history.loc[dates[t], key] = value
                            
                except Exception as e:
                    print(f"Optimization failed at {dates[t]}: {str(e)}")
                    # Keep current weights if optimization fails
                    metrics_history.loc[dates[t]] = np.nan
            
            # Store weights
            portfolio_weights.iloc[t] = current_weights
            
            # Calculate portfolio return
            portfolio_returns.iloc[t] = (
                returns.iloc[t] @ current_weights
            )
        
        # Calculate backtest metrics
        backtest_metrics = self._calculate_backtest_metrics(
            portfolio_returns=portfolio_returns,
            portfolio_weights=portfolio_weights,
            metrics_history=metrics_history
        )
        
        return {
            'returns': portfolio_returns,
            'weights': portfolio_weights,
            'metrics_history': metrics_history,
            'backtest_metrics': backtest_metrics
        }
    
    def _calculate_backtest_metrics(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights: pd.DataFrame,
        metrics_history: pd.DataFrame
    ) -> pd.Series:
        """Calculate comprehensive backtest performance metrics"""
        # Basic return metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (12 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(12)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility
        
        # Drawdown analysis
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Risk metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        sortino_ratio = (annualized_return - self.risk_free_rate) / (downside_returns.std() * np.sqrt(252))
        
        # Turnover analysis
        weight_changes = portfolio_weights.diff().abs().sum(axis=1)
        avg_turnover = weight_changes.mean()
        
        # Trading metrics
        total_transaction_costs = (weight_changes * self.transaction_cost).sum()
        net_return = total_return - total_transaction_costs
        
        return pd.Series({
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Maximum Drawdown': max_drawdown,
            'Average Turnover': avg_turnover,
            'Total Transaction Costs': total_transaction_costs,
            'Net Return': net_return
        })
    
    def plot_backtest_results(self, results: Dict[str, Union[pd.Series, pd.DataFrame]]):
        """Create comprehensive visualization of backtest results"""
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2)
        
        # 1. Cumulative Returns
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_cumulative_returns(results['returns'], ax1)
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_drawdown(results['returns'], ax2)
        
        # 3. Rolling Metrics
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_rolling_metrics(results['returns'], ax3)
        
        # 4. Weight Evolution
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_weight_evolution(results['weights'], ax4)
        
        # 5. Risk Contribution
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_risk_contribution(results['weights'], ax5)
        
        plt.tight_layout()
        plt.show()
        
    def _plot_cumulative_returns(self, returns: pd.Series, ax: plt.Axes):
        """Plot cumulative returns with key metrics"""
        cumulative_returns = (1 + returns).cumprod()
        cumulative_returns.plot(ax=ax, label='Portfolio')
        
        # Add benchmark if available
        if hasattr(self, 'benchmark_returns'):
            benchmark_cum_returns = (1 + self.benchmark_returns).cumprod()
            benchmark_cum_returns.plot(ax=ax, label='Benchmark', alpha=0.7)
            
        ax.set_title('Cumulative Returns')
        ax.legend()
        ax.grid(True)
        
    def _plot_drawdown(self, returns: pd.Series, ax: plt.Axes):
        """Plot drawdown chart"""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        
        drawdowns.plot(ax=ax, kind='area', alpha=0.7)
        ax.set_title('Drawdown')
        ax.grid(True)
        
    def _plot_rolling_metrics(self, returns: pd.Series, ax: plt.Axes):
        """Plot rolling Sharpe ratio and volatility"""
        window = 12  # 1 year rolling window
        
        rolling_vol = returns.rolling(window).std() * np.sqrt(12)
        rolling_ret = returns.rolling(window).mean() * 12
        rolling_sharpe = rolling_ret / rolling_vol
        
        rolling_sharpe.plot(ax=ax, label='Rolling Sharpe')
        rolling_vol.plot(ax=ax, label='Rolling Volatility', secondary_y=True)
        
        ax.set_title('Rolling Metrics')
        ax.legend()
        ax.grid(True)
        
    def _plot_weight_evolution(self, weights: pd.DataFrame, ax: plt.Axes):
        """Plot portfolio weight evolution"""
        weights.plot(ax=ax, kind='area', stacked=True)
        ax.set_title('Portfolio Composition')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
    def _plot_risk_contribution(self, weights: pd.DataFrame, ax: plt.Axes):
        """Plot risk contribution evolution"""
        # Calculate risk contributions through time
        risk_contributions = pd.DataFrame(index=weights.index, columns=weights.columns)
        
        for t in range(len(weights)):
            w = weights.iloc[t].values
            marginal_risk = self.covariance @ w
            contrib = w * marginal_risk
            risk_contributions.iloc[t] = contrib / contrib.sum()
            
        risk_contributions.plot(ax=ax, kind='area', stacked=True)
        ax.set_title('Risk Contribution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)