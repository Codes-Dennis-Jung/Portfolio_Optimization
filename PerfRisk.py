from typing import Union, Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats
import statsmodels.api as sm
from scipy.stats import norm, t, jarque_bera
import warnings

def __check_clean_data(
    data: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """Check and clean input data"""
    if isinstance(data, pd.Series):
        return data.to_frame()
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise TypeError("Input data must be a pandas DataFrame or Series!")

def __scaling(
    scale: str
) -> int:
    """Convert time scale to numerical factor"""
    scales = {
        "daily": 252,
        "weekly": 52,
        "monthly": 12,
        "quarterly": 4,
        "yearly": 1
    }
    try:
        return scales[scale]
    except KeyError:
        raise ValueError("Please insert correct scaling!")

def __percent(
    percent: bool,
    series: Union[pd.Series, pd.DataFrame, np.ndarray]
) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
    """Convert to percentage if needed"""
    return series * 100 if percent else series

def return_calculate(
    data: Union[pd.DataFrame, pd.Series],
    discret: bool = True,
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute discrete or continuous returns
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Input price data
    discret : bool
        If True, compute discrete returns, else log returns
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Calculated returns
    """
    df_ = __check_clean_data(data)
    if discret:
        calculate_return = (df_ / df_.shift(1)) - 1
        calculate_return.iloc[0,:] = 0
    else:
        calculate_return = np.log(df_).diff()
        calculate_return.iloc[0,:] = 0
    return __percent(percent, calculate_return)

def return_annualized(
    data: Union[pd.DataFrame, pd.Series],
    scale: str = "monthly",
    geometric: bool = True,
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute annualized returns
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    scale : str
        Time scale ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
    geometric : bool
        If True, use geometric average, else arithmetic
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Annualized returns
    """
    df_ = __check_clean_data(data)
    sc_ = __scaling(scale)
    n = len(df_)
    
    if geometric:
        ann_ret = (df_ + 1).prod(axis=0, skipna=True) ** (sc_ / n) - 1
    else:
        ann_ret = df_.mean(axis=0, skipna=True) * sc_
        
    an_ret = pd.DataFrame(
        __percent(percent, ann_ret),
        index=df_.columns,
        columns=['Annualized Return']
    )
    return an_ret

def return_annualized_excess(
    data: Union[pd.DataFrame, pd.Series],
    bmk: Union[pd.DataFrame, pd.Series],
    scale: str = "monthly",
    geometric: bool = True,
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute annualized excess returns relative to benchmark
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Portfolio returns
    bmk : Union[pd.DataFrame, pd.Series]
        Benchmark returns (single or multiple)
    scale : str
        Time scale ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
    geometric : bool
        If True, use geometric averaging
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Annualized excess returns
    """
    df_ = __check_clean_data(data)
    bm_ = __check_clean_data(bmk)
    
    # Check lengths match
    if len(df_.index) != len(bm_.index):
        raise ValueError("Length of DataFrames must be equal!")
    
    # Handle single benchmark case
    if bm_.shape[1] == 1 and df_.shape[1] > 1:
        bm_ = pd.concat([bm_] * df_.shape[1], axis=1)
        bm_.columns = df_.columns
    
    # Calculate annualized returns
    df_ret = return_annualized(df_, scale, geometric, percent=False)
    bm_ret = return_annualized(bm_, scale, geometric, percent=False)
    
    # Calculate excess returns
    if geometric:
        excess_returns = (1 + df_ret.values) / (1 + bm_ret.values) - 1
    else:
        excess_returns = df_ret.values - bm_ret.values
    
    # Create output DataFrame
    result = pd.DataFrame(
        __percent(percent, excess_returns),
        index=df_.columns,
        columns=['Annualized Excess Return']
    )
    
    return result

def volatility_annualized(
    data: Union[pd.DataFrame, pd.Series],
    scale: str = "monthly",
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute annualized volatility
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    scale : str
        Time scale
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Annualized volatility
    """
    df_ = __check_clean_data(data)
    sc_ = __scaling(scale)
    
    annualized_volatility = np.sqrt(sc_) * df_.std(
        ddof=0,
        axis=0,
        skipna=True,
        numeric_only=True
    )
    
    return pd.DataFrame(
        __percent(percent, annualized_volatility),
        index=annualized_volatility.index,
        columns=['Annualized Volatility']
    )

def sharpe_ratio(
    data: Union[pd.DataFrame, pd.Series],
    rfr: Union[pd.DataFrame, pd.Series, float],
    geometric: bool = True,
    scale: str = "monthly",
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute Sharpe ratio
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    rfr : Union[pd.DataFrame, pd.Series, float]
        Risk-free rate
    geometric : bool
        If True, use geometric returns
    scale : str
        Time scale
    percent : bool
        If True, use percentage values
        
    Returns
    -------
    pd.DataFrame
        Sharpe ratios
    """
    df_vol = volatility_annualized(data, scale, percent)
    exc_ret = return_annualized_excess(data, rfr, scale, geometric, percent)
    
    return pd.DataFrame(
        exc_ret.values / df_vol.values,
        index=data.columns,
        columns=['Sharpe Ratio']
    )

def beta_coeff(
    y: Union[pd.DataFrame, pd.Series],
    x: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """
    Compute beta coefficient with significance stars
    
    Parameters
    ----------
    y : Union[pd.DataFrame, pd.Series]
        Dependent variable
    x : Union[pd.DataFrame, pd.Series]
        Independent variable
        
    Returns
    -------
    pd.DataFrame
        Beta coefficients with significance stars
    """
    y = __check_clean_data(y)
    x = __check_clean_data(x)
    
    if len(y) < 2 or len(x) < 2:
        msg = "Beta is not well-defined with less than two samples."
        warnings.warn(msg)
        return float('nan')
        
    lst: List[pd.DataFrame] = []
    for k in range(len(list(x))):
        ind = x.iloc[:,[k]]
        exog = sm.add_constant(ind)
        nwlag = compute_newey_west_lag(n=len(ind), method='stock-watson')
        model = sm.OLS(y, exog, missing='drop', hasconst=True)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': nwlag})
        
        beta_coeff = pd.DataFrame(
            results.params[1],
            index=ind.columns,
            columns=['Beta coefficient']
        ).round(3)
        
        pval = pd.DataFrame(
            results.pvalues[1],
            index=ind.columns,
            columns=['P-value']
        ).round(3)
        
        lst.append(pd.concat([beta_coeff, pval], axis=1))
    
    res = pd.concat(lst, axis=0)
    beta = res.iloc[:,[0]]
    sign = res.iloc[:,[1]]
    
    sign = sign.where(cond=sign > 0.01, other=77)
    sign = sign.where(cond=sign > 0.05, other=88)
    sign = sign.where(cond=sign > 0.1, other=99)
    sign = sign.where(cond=sign >= 77, other=1)
    sign = sign.replace(99,'*').replace(88,'**').replace(77,'***').replace(1,'')
    
    sig_beta = pd.concat([beta.round(2), sign], axis=1)
    sig_beta.index = [y.columns]
    return sig_beta

def drawdowns(
    data: Union[pd.DataFrame, pd.Series],
    geometric: bool = True,
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute drawdown time series
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    geometric : bool
        If True, use geometric returns
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Drawdown series
    """
    df_ = __check_clean_data(data)
    
    if geometric:
        cumulative_ret = (1 + data.fillna(0)).cumprod(skipna=True)
        cumulative_ret_max = cumulative_ret.cummax(skipna=True)
        drawdowns = cumulative_ret.div(cumulative_ret_max, axis=0) - 1
        drawdowns.iloc[0,:] = 0
    else:
        cumulative_ret = data.fillna(0).cumsum() + 1
        cumulative_ret_max = cumulative_ret.cummax(skipna=True)
        drawdowns = cumulative_ret.div(cumulative_ret_max, axis=0) - 1
        drawdowns.iloc[0,:] = 0
        
    return pd.DataFrame(
        __percent(percent, drawdowns),
        index=df_.index,
        columns=df_.columns
    )

def max_drawdowns(
    data: Union[pd.DataFrame, pd.Series],
    percent: bool = True,
    geometric: bool = True
) -> pd.DataFrame:
    """
    Compute maximum drawdown
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    percent : bool
        If True, return percentage values
    geometric : bool
        If True, use geometric returns
        
    Returns
    -------
    pd.DataFrame
        Maximum drawdowns
    """
    df_ = __check_clean_data(data)
    dd_max = drawdowns(df_, geometric, percent).min(axis=0, skipna=True)
    
    return pd.DataFrame(
        dd_max,
        index=dd_max.index,
        columns=['Maximum Drawdown']
    )

def signal_to_noise_ratio(
    data: Union[pd.DataFrame, pd.Series],
    signal: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """
    Compute signal to noise ratio
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    signal : Union[pd.DataFrame, pd.Series]
        Signal data
        
    Returns
    -------
    pd.DataFrame
        Signal to noise ratios
    """
    df_ = __check_clean_data(data)
    sig_ = __check_clean_data(signal)
    
    if len(df_.index) != len(sig_.index):
        raise ValueError("Length of DataFrames must be equal!")
        
    lst: List[pd.Series] = []
    for i in range(len(list(df_))):
        nom = df_.iloc[:,[i]].corrwith(
            method='pearson',
            other=sig_.iloc[:,[i]],
            axis=0
        )
        denom = np.sqrt(1 - nom.values**2)
        SNR = nom.values / denom
        lst.append(SNR)
        
    res = pd.concat(lst, axis=1)
    res.columns = df_.columns
    res.index = ['Signal to Noise']
    return res

def dm_test(
    actual_df: Union[pd.DataFrame, pd.Series],
    pred1_df: Union[pd.DataFrame, pd.Series],
    pred2_df: Union[pd.DataFrame, pd.Series],
    h: int,
    crit: str,
    power: Optional[float] = None
) -> float:
    """
    Compute Diebold-Mariano test
    
    Parameters
    ----------
    actual_df : Union[pd.DataFrame, pd.Series]
        Actual values
    pred1_df : Union[pd.DataFrame, pd.Series]
        First set of predictions
    pred2_df : Union[pd.DataFrame, pd.Series]
        Second set of predictions
    h : int
        Forecast horizon
    crit : str
        Criterion ('MSE', 'MAD', 'MAPE', 'poly')
    power : Optional[float]
        Power for polynomial criterion
        
    Returns
    -------
    float
        P-value from DM test
    """
    actual_lst = __check_clean_data(actual_df).iloc[:,[0]].squeeze('columns').tolist()
    pred1_lst = __check_clean_data(pred1_df).iloc[:,[0]].squeeze('columns').tolist()
    pred2_lst = __check_clean_data(pred2_df).iloc[:,[0]].squeeze('columns').tolist()
    
    e1_lst: List[float] = []
    e2_lst: List[float] = []
    d_lst: List[float] = []
    T = float(len(actual_lst))
    
    if crit == "MSE":
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
    elif crit == "MAD":
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
    elif crit == "MAPE":
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
    elif crit == "poly":
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
            
    for e1, e2 in zip(e1_lst, e2_lst):
        d_lst.append(e1 - e2)
        
    mean_d = pd.Series(d_lst).mean()
    
    def autocovariance(
        Xi: List[float],
        N: int,
        k: int,
        Xs: float
    ) -> float:
        autoCov = 0.0
        T = float(N)
        for i in np.arange(0, N-k):
            autoCov += ((Xi[i+k]) - Xs)*(Xi[i] - Xs)
        return (1/T)*autoCov
    
    gamma: List[float] = []
    for lag in range(0, h):
        gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))
        
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat = V_d**(-0.5)*mean_d
    harvey_adj = ((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    
    return 2*t.cdf(-abs(DM_stat), df=T-1)

def skewness_annualized(
    data: Union[pd.DataFrame, pd.Series],
    scale: str = 'monthly'
) -> pd.DataFrame:
    """
    Compute annualized skewness
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    scale : str
        Time scale
        
    Returns
    -------
    pd.DataFrame
        Annualized skewness
    """
    df_ = __check_clean_data(data)
    sc_ = __scaling(scale)
    
    skew_an = df_.skew(skipna=True)/np.sqrt([sc_])
    skew_an = skew_an.to_frame()
    skew_an.columns = ['Skewness p.a.']
    return skew_an

def kurtosis_annualized(
    data: Union[pd.DataFrame, pd.Series],
    scale: str = 'monthly'
) -> pd.DataFrame:
    """
    Compute annualized kurtosis
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    scale : str
        Time scale
        
    Returns
    -------
    pd.DataFrame
        Annualized kurtosis
    """
    if scale != 'monthly':
        print('Scaling needs to be adjusted!')
        
    df_ = __check_clean_data(data)
    kurt_an = df_.kurt(skipna=True)/12 + 11/4
    kurt_an = kurt_an.to_frame()
    kurt_an.columns = ['Kurtosis p.a.']
    return kurt_an

def adjusted_sharpe_ratio_annualized(
    data: Union[pd.DataFrame, pd.Series],
    scale: str = 'monthly'
) -> pd.DataFrame:
    """
    Compute sharpe ratio adjusted by kurtosis and skewness
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    scale : str
        Time scale
        
    Returns
    -------
    pd.DataFrame
        Adjusted Sharpe ratio
    """
    ret = return_annualized(data, scale, geometric=True, percent=True)
    vol = volatility_annualized(data, scale, percent=True)
    sr = pd.DataFrame(
        ret.values/vol.values,
        index=data.columns,
        columns=["Adjusted Sharpe Ratio"]
    )
    k = data.kurt().to_frame().values
    sk = data.skew().to_frame().values
    ASR = sr * (1 + (sk/6)*sr - ((k-3)/24)*sr**2)
    return ASR

def absolute_sum_correlation(
    data: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """
    Compute absolute sum of correlations
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
        
    Returns
    -------
    pd.DataFrame
        Absolute sum of correlations
    """
    data = data.fillna(0)
    lst_: List[pd.DataFrame] = []
    
    for i in range(len(list(data))):
        a = data.iloc[:,[i]]
        lst: List[float] = []
        b_ = data.drop(a.columns.tolist(), axis=1)
        
        for k in range(len(list(b_))):
            pair = pd.concat([a,b_.iloc[:,[k]]], axis=1)
            lst.append(pair.corr().iloc[1,0])
            
        val = pd.DataFrame(
            np.sum(np.absolute(np.array(lst))),
            index=a.columns,
            columns=['Abs. Sum of pairwise Correl']
        )
        lst_.append(val)
        
    res = pd.concat(lst_, axis=0).sort_values(
        by=['Abs. Sum of pairwise Correl'],
        ascending=False
    ).T
    res['Total average'] = res.mean(axis=1)
    return res.T

def portfolio_vol(
    weights: np.ndarray,
    covmat: np.ndarray
) -> float:
    """
    Compute portfolio volatility
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    covmat : np.ndarray
        Covariance matrix
        
    Returns
    -------
    float
        Portfolio volatility
    """
    return float((weights.T @ covmat @ weights)**0.5)

def risk_contribution(
    weights: Union[pd.DataFrame, pd.Series],
    data: Union[pd.DataFrame, pd.Series],
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute risk contribution
    
    Parameters
    ----------
    weights : Union[pd.DataFrame, pd.Series]
        Portfolio weights
    data : Union[pd.DataFrame, pd.Series]
        Return data
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Risk contributions
    """
    df_ = __check_clean_data(data)
    w_ = __check_clean_data(weights).T
    cov_ = df_.cov()
    
    total_portfolio_var = portfolio_vol(w_, cov_)**2
    marginal_contrib = cov_ @ w_
    risk_contrib = np.multiply(marginal_contrib, w_)/total_portfolio_var.values
    
    return pd.DataFrame(
        __percent(percent, risk_contrib),
        index=df_.columns,
        columns=['Risk contribution']
    )

def calculate_portfolio_var(
    weights: np.ndarray,
    covar: np.ndarray
) -> float:
    """
    Calculate portfolio variance
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    covar : np.ndarray
        Covariance matrix
        
    Returns
    -------
    float
        Portfolio variance
    """
    return float(weights.T @ covar @ weights)

def diversification_ratio(
    weights: Union[pd.DataFrame, pd.Series],
    data: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """
    Compute diversification ratio
    
    Parameters
    ----------
    weights : Union[pd.DataFrame, pd.Series]
        Portfolio weights
    data : Union[pd.DataFrame, pd.Series]
        Return data
        
    Returns
    -------
    pd.DataFrame
        Diversification ratio
    """
    df_ = __check_clean_data(data)
    w_ = __check_clean_data(weights).T
    cov_ = df_.cov()
    
    w_vol = np.dot(np.sqrt(np.diag(cov_)), w_)
    port_vol = calculate_portfolio_var(w_, cov_)
    diversification_ratio = w_vol/port_vol
    
    return pd.DataFrame(
        diversification_ratio,
        index=['Portfolio'],
        columns=['Diversification Ratio']
    )

def zscore_scaling(
    data: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """
    Compute z-score scaling
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Input data
        
    Returns
    -------
    pd.DataFrame
        Z-score scaled data
    """
    return pd.DataFrame(
        stats.zscore(data, axis=1, nan_policy='omit'),
        columns=data.columns,
        index=data.index
    )

def scaling(
    data: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """
    Scale data using rank transformation
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Input data
        
    Returns
    -------
    pd.DataFrame
        Scaled data
    """
    lst: List[pd.DataFrame] = []
    for i in range(len(data)):
        series = data.iloc[[i],:].squeeze()
        try:
            temp2 = np.squeeze(series.rank(
                method='first',
                numeric_only=True
            ))
            series.loc[~series.isna()] = (temp2/(1+len(temp2)) - 0.5)
        except ValueError:
            raise ValueError("CHECK DATA")
        lst.append(pd.DataFrame(series).T)
    data = pd.concat(lst)
    data = data.replace([np.inf, -np.inf], np.nan)
    return data

def performance_risk_table(
    data: Union[pd.DataFrame, pd.Series],
    bmk: Union[pd.DataFrame, pd.Series],
    rfr: Union[pd.DataFrame, pd.Series, float],
    scale: str = "monthly",
    geometric: bool = True,
    percent: bool = True,
    cutoff: float = 0.05
) -> pd.DataFrame:
    """
    Create comprehensive performance and risk metrics table
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    bmk : Union[pd.DataFrame, pd.Series]
        Benchmark returns
    rfr : Union[pd.DataFrame, pd.Series, float]
        Risk-free rate
    scale : str
        Time scale
    geometric : bool
        If True, use geometric returns
    percent : bool
        If True, use percentage values
    cutoff : float
        Cutoff for VaR calculation
        
    Returns
    -------
    pd.DataFrame
        Performance and risk metrics table
    """
    return pd.concat([
        return_annualized(data, scale, geometric, percent),
        return_annualized_excess(data, bmk, scale, geometric, percent),
        max_drawdowns(data, percent, geometric),
        volatility_annualized(data, scale, percent),
        semideviation(data, scale, percent),
        tracking_error(data, bmk, scale, percent),
        information_ratio(data, bmk, scale, geometric, percent),
        sharpe_ratio(data, rfr, geometric, scale, percent),
        sortino_ratio(data, rfr, scale),
        calmar_ratio(data, scale, geometric, percent),
        appraisal_ratio(data, bmk, geometric, scale, percent),
        appraisal_ratio_regress(data, bmk),
        return_risk_ratio_annualized(data, scale),
        risk_adj_abn_ret(data, bmk, geometric, scale, percent),
        value_at_risk_cornish_fisher(data, cutoff, percent),
        conditional_value_at_risk(data, cutoff, percent),
        adjusted_sharpe_ratio_annualized(data, scale)
    ], axis=1)

def value_at_risk(
    data: Union[pd.DataFrame, pd.Series],
    cutoff: float = 0.05,
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute Value at Risk
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    cutoff : float
        VaR confidence level
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Value at Risk
    """
    df_ = __check_clean_data(data)
    n = len(df_.columns)
    var = pd.DataFrame(
        [np.percentile(df_.iloc[:,[x]], 100*cutoff) for x in range(n)],
        columns=['VaR {}'.format((1-cutoff)*100)],
        index=df_.columns
    )
    return __percent(percent, var)

def value_at_risk_cornish_fisher(
    data: Union[pd.DataFrame, pd.Series],
    cutoff: float = 0.05,
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute VaR with Cornish-Fisher expansion
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    cutoff : float
        VaR confidence level
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Modified VaR
    """
    df_ = __check_clean_data(data)
    k = df_.kurt()
    s = df_.skew()
    z = norm.ppf(cutoff)
    z = (z + (z**2-1)*s/6 + (z**3-3*z)*(k-3)/24 - (2*z**3-5*z)*(s**2)/36)
    var = -(df_.mean() + z*df_.std(ddof=0))
    var = var.to_frame()
    var.columns = ['VaR (Cornish Fisher) {}'.format((1-cutoff)*100)]
    return __percent(percent, -1*var)

def value_at_risk_gaussian(
    data: Union[pd.DataFrame, pd.Series],
    cutoff: float = 0.05,
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute parametric Gaussian VaR
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    cutoff : float
        VaR confidence level
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Gaussian VaR
    """
    df_ = __check_clean_data(data)
    z = norm.ppf(cutoff)
    var = -(df_.mean() + z*df_.std(ddof=0))
    var = var.to_frame()
    var.columns = ['VaR (Gaussian) {}'.format((1-cutoff)*100)]
    return __percent(percent, -1*var)

def is_normal(
    data: Union[pd.DataFrame, pd.Series],
    level: float = 0.05
) -> bool:
    """
    Test for normality using Jarque-Bera test
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    level : float
        Significance level
        
    Returns
    -------
    bool
        True if normal at given significance level
    """
    _, p_value = jarque_bera(data)
    return bool(p_value > level)

def conditional_value_at_risk(
    data: Union[pd.DataFrame, pd.Series],
    cutoff: float = 0.05,
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute Conditional Value at Risk (Expected Shortfall)
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    cutoff : float
        CVaR confidence level
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Conditional Value at Risk
    """
    df_ = __check_clean_data(data)
    n = len(df_.columns)
    lst: List[float] = []
    
    for i in range(0, n):
        df_in = df_.iloc[:,i]
        cutoff_index = int((len(df_in)-1)*cutoff)
        cvar = np.mean(np.partition(df_in, cutoff_index)[:cutoff_index+1])
        lst.append(cvar)
        
    cvar = pd.DataFrame(
        lst,
        columns=['CVaR {}%'.format((1-cutoff)*100)],
        index=df_.columns
    )
    return __percent(percent, cvar)

def information_coefficient(
    data: Union[pd.DataFrame, pd.Series],
    signal: Union[pd.DataFrame, pd.Series],
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute Information Coefficient
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    signal : Union[pd.DataFrame, pd.Series]
        Signal data
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Information coefficients
    """
    df_ = __check_clean_data(data)
    sig_ = __check_clean_data(signal)
    
    if len(df_.index) != len(sig_.index):
        raise ValueError("Length of DataFrames must be equal!")
        
    ic = pd.DataFrame(
        [stats.spearmanr(df_.iloc[:,x], sig_.iloc[:,x])[0] 
         for x in range(len(df_.columns))],
        index=df_.columns,
        columns=['Information Coefficient']
    )
    return __percent(percent, ic)

def omega_ratio(
    data: Union[pd.DataFrame, pd.Series],
    rfr: float = 0.0,
    required_return: Union[float, pd.DataFrame, pd.Series] = 0.0,
    scale: str = "monthly"
) -> pd.DataFrame:
    """
    Compute Omega ratio
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    rfr : float
        Risk-free rate
    required_return : Union[float, pd.DataFrame, pd.Series]
        Minimum acceptable return
    scale : str
        Time scale
        
    Returns
    -------
    pd.DataFrame
        Omega ratios
    """
    df_ = __check_clean_data(data)
    sc_ = __scaling(scale)
    risk_free = rfr
    n = len(list(df_))
    
    if required_return <= -1:
        return np.nan
    elif isinstance(required_return, (pd.DataFrame, pd.Series)):
        return_threshold = (1 + required_return)**(1./sc_) - 1
    else:
        return_threshold = required_return
        
    returns_less_thresh = df_ - risk_free - return_threshold
    lst: List[pd.DataFrame] = []
    
    for i in range(n):
        numer = returns_less_thresh.iloc[:,[i]][
            returns_less_thresh.iloc[:,[i]] > 0.0
        ].sum()
        denom = -1.0 * returns_less_thresh.iloc[:,[i]][
            returns_less_thresh.iloc[:,[i]] < 0.0
        ].sum()
        lst.append(pd.DataFrame(numer/denom, columns=["Omega Ratio"]))
        
    return pd.concat(lst, axis=0)

def sortino_ratio(
    data: Union[pd.DataFrame, pd.Series],
    rfr: Union[pd.DataFrame, pd.Series, float],
    scale: str = "monthly"
) -> pd.DataFrame:
    """
    Compute Sortino ratio
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    rfr : Union[pd.DataFrame, pd.Series, float]
        Risk-free rate
    scale : str
        Time scale
        
    Returns
    -------
    pd.DataFrame
        Sortino ratios
    """
    df_ = __check_clean_data(data)
    sc_ = __scaling(scale)
    n = len(list(df_))
    down_diff = df_ - rfr.values.reshape(-1,1)
    lst: List[pd.DataFrame] = []
    
    for i in range(n):
        df = down_diff.iloc[:,[i]]
        down_dev = (df[df < df.mean()].std(
            axis=0,
            ddof=0,
            skipna=True,
            numeric_only=True
        ) * np.sqrt(sc_)).values
        
        ret = return_annualized(
            df,
            scale,
            geometric=False,
            percent=False
        ).values
        
        lst.append(pd.DataFrame(
            ret/down_dev,
            index=df_.iloc[:,[i]].columns,
            columns=["Sortino Ratio"]
        ))
    
    return pd.concat(lst, axis=0)

def treynor_ratio(
    data: Union[pd.DataFrame, pd.Series],
    bmk: Union[pd.DataFrame, pd.Series],
    geometric: bool = True,
    percent: bool = True,
    scale: str = "monthly"
) -> pd.DataFrame:
    """
    Compute Treynor ratio
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    bmk : Union[pd.DataFrame, pd.Series]
        Benchmark returns
    geometric : bool
        If True, use geometric returns
    percent : bool
        If True, return percentage values
    scale : str
        Time scale
        
    Returns
    -------
    pd.DataFrame
        Treynor ratios
    """
    df_ = __check_clean_data(data)
    bm_ = __check_clean_data(bmk)
    
    if len(df_.index) != len(bm_.index):
        raise ValueError("Length of DataFrames must be equal!")
        
    ret = return_annualized_excess(data, bmk, scale, geometric, percent)
    lst: List[float] = []
    
    for i in range(len(list(df_))):
        nwlag = compute_newey_west_lag(n=len(df_), method='stock-watson')
        if len(list(bm_)) > 1:
            beta = sm.OLS(
                df_.iloc[:,[i]],
                sm.add_constant(bm_.iloc[:,[i]])
            ).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': nwlag}
            ).params[-1]
        else:
            beta = sm.OLS(
                df_.iloc[:,[i]],
                sm.add_constant(bm_)
            ).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': nwlag}
            ).params[-1]
        lst.append(beta)
        
    return pd.DataFrame(
        ret.values/np.array(lst).reshape(-1,1),
        index=df_.columns,
        columns=["Treynor Ratio"]
    )

def equal_length(
    df1: pd.DataFrame,
    df2: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure two dataframes have matching indices
    
    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe
    df2 : pd.DataFrame
        Second dataframe
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Aligned dataframes
    """
    df2 = df2[df2.index.isin(df1.index.to_list())]
    df1 = df1[df1.index.isin(df2.index.to_list())]
    return df1, df2

def appraisal_ratio(
    data: Union[pd.DataFrame, pd.Series],
    bmk: Union[pd.DataFrame, pd.Series],
    geometric: bool = True,
    scale: str = "monthly",
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute Appraisal ratio (alpha/tracking error)
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Portfolio returns
    bmk : Union[pd.DataFrame, pd.Series]
        Benchmark returns
    geometric : bool
        If True, use geometric returns
    scale : str
        Time scale for annualization
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Appraisal ratios
    """
    # Calculate excess returns
    exc_ret = return_annualized_excess(
        data, bmk,
        scale=scale,
        geometric=geometric,
        percent=percent
    )
    
    # Calculate tracking error
    te = tracking_error(
        data, bmk,
        scale=scale,
        percent=percent
    )
    
    # Calculate appraisal ratio
    app_r = pd.DataFrame(
        exc_ret.values / te.values,
        index=data.columns,
        columns=['Appraisal Ratio']
    )
    
    return app_r

def appraisal_ratio_regress(
    data: Union[pd.DataFrame, pd.Series],
    bmk: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """
    Compute regression-based Appraisal ratio
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    bmk : Union[pd.DataFrame, pd.Series]
        Benchmark returns
        
    Returns
    -------
    pd.DataFrame
        Appraisal ratios
    """
    df_ = __check_clean_data(data)
    bm_ = __check_clean_data(bmk)
    df_, bm_ = equal_length(df_.dropna(axis=0), bm_)
    
    if len(df_.index) != len(bm_.index):
        raise ValueError("Length of DataFrames must be equal!")
        
    lst: List[pd.DataFrame] = []
    nwlag = compute_newey_west_lag(n=len(df_), method='stock-watson')
    for i in range(len(list(df_))):
        if len(list(bm_)) > 1:
            lst.append(
                sm.OLS(
                    df_.iloc[:,[i]],
                    sm.add_constant(bm_.iloc[:,[i]])
                ).fit(
                    cov_type='HAC',
                    cov_kwds={'maxlags': nwlag}
                ).params[-1]
            )
        else:
            res = sm.OLS(
                df_.iloc[:,[i]],
                sm.add_constant(bm_)
            ).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': nwlag}
            )
            alpha = pd.DataFrame(res.params).T.iloc[:,[0]].values
            se = pd.DataFrame(res.bse).T.iloc[:,[0]].values
            lst.append(pd.DataFrame(alpha/se, columns=[i]))
            
    res = pd.concat(lst, axis=1)
    res.columns = df_.columns
    res.index = ['Appraisal Ratio (alpha/se)']
    return res.T

def hit_ratio(
    data: Union[pd.DataFrame, pd.Series],
    bmk: Union[pd.DataFrame, pd.Series],
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute hit ratio
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    bmk : Union[pd.DataFrame, pd.Series]
        Benchmark returns
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Hit ratios
    """
    df_ = __check_clean_data(data)
    bm_ = __check_clean_data(bmk)
    
    if len(df_.index) != len(bm_.index):
        raise ValueError("Length of DataFrames must be equal!")
        
    df_dir = pd.DataFrame(np.sign(df_).values).replace(0,1)
    bmk_dir = pd.DataFrame(np.sign(bm_).values).replace(0,1)
    
    n = len(df_.index)
    hr = pd.DataFrame(
        np.where((df_dir-bmk_dir)!=0, 0, 1),
        index=df_.index,
        columns=df_.columns
    ).sum()/n
    
    return pd.DataFrame(
        __percent(percent, hr),
        columns=['Hit Ratio']
    )

def calmar_ratio(
    data: Union[pd.DataFrame, pd.Series],
    scale: str = "monthly",
    geometric: bool = True,
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute Calmar ratio
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    scale : str
        Time scale
    geometric : bool
        If True, use geometric returns
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Calmar ratios
    """
    ret = return_annualized(data, scale, geometric, percent)
    dd = max_drawdowns(data, geometric, percent)
    
    return pd.DataFrame(
        ret.values/abs(dd.values),
        columns=['Calmar Ratio'],
        index=data.columns
    )

def tracking_error(
    data: Union[pd.DataFrame, pd.Series],
    bmk: Union[pd.DataFrame, pd.Series],
    scale: str = "monthly",
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute tracking error (standard deviation of excess returns)
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Portfolio returns
    bmk : Union[pd.DataFrame, pd.Series]
        Benchmark returns (single or multiple)
    scale : str
        Time scale for annualization
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Tracking error values
    """
    # Convert inputs to DataFrames
    df_ = __check_clean_data(data)
    bm_ = __check_clean_data(bmk)
    sc_ = __scaling(scale)
    
    # Check index alignment
    if not df_.index.equals(bm_.index):
        raise ValueError("DataFrames must have identical indices!")
    
    # Handle single benchmark case
    if bm_.shape[1] == 1 and df_.shape[1] > 1:
        bm_values = np.repeat(bm_.values, df_.shape[1], axis=1)
    else:
        bm_values = bm_.values
        
    # Calculate excess returns
    excess_returns = df_.values - bm_values
    
    # Convert to DataFrame with proper columns
    excess_df = pd.DataFrame(
        excess_returns,
        index=df_.index,
        columns=df_.columns
    )
    
    # Calculate tracking error (annualized standard deviation of excess returns)
    error_tracking = np.sqrt(sc_) * excess_df.std(
        ddof=1,  # Using n-1 for sample standard deviation
        axis=0,
        skipna=True
    )
    
    # Format result
    tracking_error_df = pd.DataFrame(
        __percent(percent, error_tracking),
        index=df_.columns,
        columns=['Tracking Error']
    )
    
    return tracking_error_df

def information_ratio(
    data: Union[pd.DataFrame, pd.Series],
    bmk: Union[pd.DataFrame, pd.Series],
    scale: str = "monthly",
    geometric: bool = True,
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute information ratio
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Portfolio returns
    bmk : Union[pd.DataFrame, pd.Series]
        Benchmark returns
    scale : str
        Time scale
    geometric : bool
        If True, use geometric returns
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Information ratios
    """
    if geometric:
        ret_exc = return_annualized_excess(
            data, bmk, scale,
            geometric=True, percent=True
        )
        te = tracking_error(data, bmk, scale, percent=True)
    else:
        ret_exc = return_annualized_excess(
            data, bmk, scale,
            geometric=False, percent=True
        )
        te = tracking_error(data, bmk, scale, percent=True)
        
    ratio_information = pd.DataFrame(
        ret_exc.values/te.values,
        index=data.columns,
        columns=['Information Ratio']
    )
    return ratio_information.fillna('Error')

def return_risk_ratio_annualized(
    data: Union[pd.DataFrame, pd.Series],
    scale: str = "monthly"
) -> pd.DataFrame:
    """
    Compute annualized return to risk ratio
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    scale : str
        Time scale
        
    Returns
    -------
    pd.DataFrame
        Return to risk ratios
    """
    ret = return_annualized(data, scale, geometric=True, percent=True)
    vol = volatility_annualized(data, scale, percent=True)
    
    return pd.DataFrame(
        ret.values/vol.values,
        index=data.columns,
        columns=['Return to Risk Ratio']
    )

def turnover_avg(
    weights: Union[pd.DataFrame, pd.Series],
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute average portfolio turnover
    
    Parameters
    ----------
    weights : Union[pd.DataFrame, pd.Series]
        Portfolio weights over time
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Average turnover
    """
    df_ = __check_clean_data(weights)
    to = (df_ - df_.shift(1))
    to.iloc[0,:] = 0
    to = to.abs().sum(axis=1).mean(axis=0, skipna=True) * 100
    
    return pd.DataFrame(
        __percent(percent, to),
        index=['Portfolio'],
        columns=['Average Turnover']
    )

def net_return_series(
    weights: Union[pd.DataFrame, pd.Series],
    data: Union[pd.DataFrame, pd.Series],
    TC: float = 0
) -> float:
    """
    Compute net returns accounting for transaction costs
    
    Parameters
    ----------
    weights : Union[pd.DataFrame, pd.Series]
        Portfolio weights
    data : Union[pd.DataFrame, pd.Series]
        Asset returns
    TC : float
        Transaction cost rate
        
    Returns
    -------
    float
        Net return
    """
    w_ = __check_clean_data(weights)
    df_ = __check_clean_data(data)
    
    if len(df_.index) != len(w_.index):
        raise ValueError("Length of DataFrames must be equal!")
        
    to = (w_ - w_.shift(1))
    to.iloc[0,:] = 0
    rets_2 = TC * (to.abs()).dot(df_)
    rets_ = (df_.dot(w_)).sum()
    
    return float(rets_ - rets_2)

def risk_adj_abn_ret(
    data: Union[pd.DataFrame, pd.Series],
    bmk: Union[pd.DataFrame, pd.Series],
    geometric: bool = True,
    scale: str = "monthly",
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute risk-adjusted abnormal returns
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Portfolio returns
    bmk : Union[pd.DataFrame, pd.Series]
        Benchmark returns
    geometric : bool
        If True, use geometric returns
    scale : str
        Time scale
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Risk-adjusted abnormal returns
    """
    df_vol = volatility_annualized(data, scale, percent)
    bmk_vol = volatility_annualized(bmk, scale, percent)
    exc_ret = return_annualized_excess(
        data, bmk,
        scale, geometric, percent
    )
    
    return pd.DataFrame(
        exc_ret.values * bmk_vol.values / df_vol.values,
        index=data.columns,
        columns=['Risk-Adjusted Abnormal Return']
    )

def semideviation(
    data: Union[pd.DataFrame, pd.Series],
    scale: str = 'monthly',
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute annualized semideviation
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    scale : str
        Time scale
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Annualized semideviation
    """
    df_ = __check_clean_data(data)
    sc_ = __scaling(scale)
    is_negative = df_ <= 0
    
    semi_dev = df_[is_negative].std(
        ddof=0,
        axis=0,
        skipna=True,
        numeric_only=True
    ) * np.sqrt(sc_)
    
    semi_dev = semi_dev.to_frame()
    semi_dev.columns = ['Annualized Semideviation']
    
    return __percent(percent, semi_dev)

def coskewness(
    data_bmk: Union[pd.DataFrame, pd.Series],
    bias_: bool = True
) -> pd.DataFrame:
    """
    Compute coskewness
    
    Parameters
    ----------
    data_bmk : Union[pd.DataFrame, pd.Series]
        Combined data and benchmark returns
    bias_ : bool
        If True, apply bias correction
        
    Returns
    -------
    pd.DataFrame
        Coskewness values
    """
    df = __check_clean_data(data_bmk)
    v = df.values
    s1 = sigma = v.std(0, keepdims=True)
    means = v.mean(0, keepdims=True)
    v1 = v - means
    s2 = sigma**2
    v2 = v1**2
    m = v.shape[0]
    skew = pd.DataFrame(
        v2.T.dot(v1)/s2.T.dot(s1)/m,
        df.columns,
        df.columns
    )
    
    if bias_:
        skew *= ((m-1)*m)**.5/(m-2)
        
    skew = skew.iloc[[0],[-1]]
    skew.columns = ['Coskewness']
    return skew

def cokurtosis(
    data_bmk: Union[pd.DataFrame, pd.Series],
    bias_: bool = True,
    fisher: bool = True,
    variant: str = 'middle'
) -> pd.DataFrame:
    """
    Compute cokurtosis
    
    Parameters
    ----------
    data_bmk : Union[pd.DataFrame, pd.Series]
        Combined data and benchmark returns
    bias_ : bool
        If True, apply bias correction
    fisher : bool
        If True, use Fisher adjustment
    variant : str
        Type of cokurtosis ('left', 'right', 'middle')
        
    Returns
    -------
    pd.DataFrame
        Cokurtosis values
    """
    df = __check_clean_data(data_bmk)
    v = df.values
    s1 = sigma = v.std(0, keepdims=True)
    means = v.mean(0, keepdims=True)
    v1 = v - means
    s2 = sigma**2
    s3 = sigma**3
    v2 = v1**2
    v3 = v1**3
    m = v.shape[0]
    
    if variant in ['left', 'right']:
        kurt = pd.DataFrame(
            v3.T.dot(v1)/s3.T.dot(s1)/m,
            df.columns,
            df.columns
        )
        if variant == 'right':
            kurt = kurt.T
    elif variant == 'middle':
        kurt = pd.DataFrame(
            v2.T.dot(v2)/s2.T.dot(s2)/m,
            df.columns,
            df.columns
        )
        
    if bias_:
        kurt = kurt*(m**2-1)/(m-2)/(m-3)-3*(m-1)**2/(m-2)/(m-3)
    if not fisher:
        kurt += 3
        
    kurt = kurt.iloc[[0],[-1]]
    kurt.columns = ['Cokurtosis']
    return kurt

def significance_test(
    data: Union[pd.DataFrame, pd.Series],
    bmk: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """
    Compute significance test
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Portfolio returns
    bmk : Union[pd.DataFrame, pd.Series]
        Benchmark returns
        
    Returns
    -------
    pd.DataFrame
        Test results with p-values
    """
    df_ = __check_clean_data(data)
    bm_ = __check_clean_data(bmk)
    
    if len(df_.index) != len(bm_.index):
        raise ValueError("Length of DataFrames must be equal!")
    
    _Signif: List[pd.DataFrame] = []
    
    for k in range(len(list(df_))):
        f = df_.iloc[:,[k]].values - bm_.values
        nwlag = compute_newey_west_lag(n=len(f), method='stock-watson')
        x = np.ones(np.shape(f))
        model = sm.OLS(f, x, missing='drop')
        results = model.fit(
            cov_type='HAC',
            cov_kwds={'maxlags': nwlag}
        )
        r = np.zeros_like(results.params)
        r[:] = [1]
        res = results.t_test(r).pvalue
        
        _Signif.append(pd.DataFrame(
            np.array([res]),
            columns=df_.iloc[:,[k]].columns,
            index=['P-value']
        ))
        
    return pd.concat(_Signif, sort=False, axis=1)

def false_discovery_control(
    pval: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """
    Compute false discovery test using Benjamini Yekutieli method
    
    Parameters
    ----------
    pval : Union[pd.DataFrame, pd.Series]
        P-values to test
        
    Returns
    -------
    pd.DataFrame
        Controlled p-values
    """
    df_ = __check_clean_data(pval)
    res = stats.false_discovery_controls(
        df_.to_numpy(),
        method='by'
    )
    return pd.DataFrame(
        res,
        index=df_.index,
        columns=df_.columns
    )

def seriel_correlation(
    data: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """
    Compute serial correlation
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Time series data
        
    Returns
    -------
    pd.DataFrame
        Serial correlation coefficients
    """
    df_ = __check_clean_data(data)
    AutoCorr: List[pd.DataFrame] = []
    
    for i in range(len(list(df_))):
        series = df_.iloc[:,[i]].squeeze()
        ser1 = series.autocorr(lag=1)
        ser2 = series.autocorr(lag=2)
        ser3 = series.autocorr(lag=3)
        ser4 = series.autocorr(lag=4)
        ser5 = series.autocorr(lag=5)
        
        SerielCorrel = pd.DataFrame(
            [ser1, ser2, ser3, ser4, ser5],
            index=['AR(1)', 'AR(2)', 'AR(3)', 'AR(4)', 'AR(5)'],
            columns=[series.name]
        )
        AutoCorr.append(SerielCorrel)
        
    return pd.concat(AutoCorr, axis=1)

def return_cumulative(
    data: Union[pd.DataFrame, pd.Series],
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute cumulative returns
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Cumulative returns
    """
    df_ = __check_clean_data(data)
    df_.iloc[0,:] = 0
    cumulative_ret = (1 + df_).cumprod(0) - 1
    
    cum_ret = pd.DataFrame(
        __percent(percent, cumulative_ret),
        index=cumulative_ret.index,
        columns=df_.columns
    )
    
    if percent:
        cum_ret = cum_ret + 100
    else:
        cum_ret = cum_ret + 1
        
    return cum_ret

def return_cumulative_zero(
    data: Union[pd.DataFrame, pd.Series],
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute cumulative returns starting from zero
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Return data
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Cumulative returns
    """
    df_ = __check_clean_data(data)
    df_.iloc[0,:] = 0
    cumulative_ret = (1 + df_).cumprod(0) - 1
    
    return pd.DataFrame(
        __percent(percent, cumulative_ret),
        index=cumulative_ret.index,
        columns=df_.columns
    )
    
def return_excess(
    data: Union[pd.DataFrame, pd.Series],
    bmk: Union[pd.DataFrame, pd.Series],
    percent: bool = True
) -> pd.DataFrame:
    """
    Compute excess return time series relative to benchmark
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Portfolio returns time series
    bmk : Union[pd.DataFrame, pd.Series]
        Benchmark returns time series
    percent : bool
        If True, return percentage values
        
    Returns
    -------
    pd.DataFrame
        Time series of excess returns
    """
    df_ = __check_clean_data(data)
    bm_ = __check_clean_data(bmk)
    
    # Verify equal lengths
    n = len(df_.index)
    m = len(bm_.index)
    if n != m:
        raise ValueError("Length of DataFrames must be equal!")
    
    # Calculate excess returns
    excess_return = pd.DataFrame(
        df_.values - bm_.values,
        index=df_.index,
        columns=df_.columns
    )
    
    # Set first observation to zero
    excess_return.iloc[0,:] = 0
    
    # Apply percentage scaling if requested
    return pd.DataFrame(
        __percent(percent, excess_return),
        index=excess_return.index,
        columns=excess_return.columns
    )
    
def add_stars_pval(pvalue: float, decimals: int = 3) -> str:
    """Helper function to format p-value with stars"""
    if np.isnan(pvalue):
        return 'NaN'
        
    formatted = f"{pvalue:.{decimals}f}"
    if pvalue <= 0.01:
        return f"{formatted}***"
    elif pvalue <= 0.05:
        return f"{formatted}**"
    elif pvalue <= 0.10:
        return f"{formatted}*"
    return formatted

def add_stars(pvalue: float, decimals: int = 3) -> str:
    """Helper function to format p-value with stars"""
    if np.isnan(pvalue):
        return 'NaN'
    
    if pvalue <= 0.01:
        return "***"
    elif pvalue <= 0.05:
        return "**"
    elif pvalue <= 0.10:
        return "*"

def add_significance_stars(pvalues: ArrayLike, decimals: int = 3, include_table: bool = False) -> Dict[str, Union[List[str], str]]:
    """
    Add significance stars to p-values based on threshold
    
    Parameters:
    -----------
    pvalues : ArrayLike
        Array of p-values
    decimals : int
        Number of decimal places
    include_table : bool
        Whether to include legend table
        
    Returns:
    --------
    Dict with formatted p-values and optional legend
    """
    results = [add_stars(pval, decimals) for pval in pvalues]
    
    output = {'formatted_pvalues': results}
    
    if include_table:
        legend = (
            "\nSignificance Levels:\n"
            "*** p ≤ 0.01\n"
            " ** p ≤ 0.05\n"
            "  * p ≤ 0.10"
        )
        output['legend'] = legend
        
    return output


def f_pval2s(t_ret: np.ndarray, T: Union[float, np.ndarray]) -> np.ndarray:
    '''
    Compute two-sided p-values for t-statistics.
    
    Parameters:
    -----------
    t_ret : np.ndarray
        Array of t-statistics
    T : Union[float, np.ndarray]
        Degrees of freedom (sample size - 1)
        
    Returns:
    --------
    np.ndarray
        Two-sided p-values for hypothesis test H0: mu=0 vs H1: mu≠0
    '''
    t_ret = np.asarray(t_ret, dtype=float)
    nvar = len(t_ret)
    pval2s = np.full(nvar, np.nan)
    
    # Handle T properly based on its type
    if isinstance(T, (int, float)):
        T = np.repeat(float(T), nvar)
    else:
        T = np.asarray(T, dtype=float)
        if len(T) < nvar:
            T = np.repeat(T[0], nvar)
            
    pval2s = 2 * stats.t.cdf(-np.abs(t_ret), df=T-1) 
    return pval2s

def compute_newey_west_lag(n: int, method: str = 'stock-watson') -> int:
    '''
    Compute the optimal lag length for Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors.
    The function implements three different methods to determine the optimal lag length based on sample size.
    
    Parameters:
    -----------
    n : int
        Sample size (number of observations). Must be a positive integer.
    method : str, optional
        Method to compute lag length. Options are:
        - 'stock-watson': Stock and Watson (2007) method
          Formula: floor(4*(n/100)^(2/9))
          Recommended for general time series applications
        - 'greene': Greene (2008) method
          Formula: floor(n^(1/3))
          More conservative approach, generally produces longer lags
        - 'newey-west': Original Newey-West (1994) method
          Formula: floor(4*(n/100)^(2/9))
          Classic approach, identical to Stock-Watson in implementation
        Default is 'stock-watson'
    
    Returns:
    --------
    int
        Recommended lag length. Will be at least 1 and at most (n-1).
    
    Raises:
    -------
    ValueError
        - If n is not a positive integer
        - If method is not one of the allowed options
        
    Examples:
    --------
    >>> compute_newey_west_lag(100, 'stock-watson')
    4
    >>> compute_newey_west_lag(100, 'greene')
    4
    
    References:
    ----------
    - Stock, J. H., & Watson, M. W. (2007). Introduction to Econometrics.
    - Greene, W. H. (2008). Econometric Analysis.
    - Newey, W. K., & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation.
    
    Notes:
    ------
    The function automatically ensures the returned lag length is between 1 and (n-1),
    regardless of the method used. This prevents invalid lag lengths in small samples.
    '''
    if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("Sample size 'n' must be a positive integer")
        
    if method.lower() == 'stock-watson' or method.lower() == 'newey-west':
        # Stock-Watson (2007) and Newey-West (1994) method
        lag = int(np.floor(4 * (n/100)**(2/9)))
    
    elif method.lower() == 'greene':
        # Greene (2008) method
        lag = int(np.floor(n**(1/3)))
    
    else:
        raise ValueError("Method must be one of: 'stock-watson', 'greene', 'newey-west'")
    
    # Ensure lag is at least 1 and less than sample size
    lag = max(1, min(lag, n-1))
    
    return lag
