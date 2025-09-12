import pandas as pd
import numpy as np
from scipy import stats

def calculate_ci_percentage_above_threshold(ci_bounds, threshold=0.9):
        if ci_bounds is None or len(ci_bounds) != 2 or ci_bounds[0] is None or ci_bounds[1] is None:
            return None
        lower_bound, upper_bound = ci_bounds
        
        if upper_bound < threshold:
            return 0.0
        
        if lower_bound >= threshold:
            return 100.0
        
        ci_width = upper_bound - lower_bound
        
        above_threshold_width = upper_bound - max(lower_bound, threshold)
        percentage = (above_threshold_width / ci_width) * 100
        return percentage


def load_single_student_simluation_scores(df: pd.DataFrame,
                                          additional_stats: bool = True,
                                          window: int =5) -> dict:
    """ Function loads the statistics for time-series of the single student
        :param df - dataframe that contains information about single student performance across multiple simulations
        :param additional_stats - Determines if additional statistics will be calculated
        :param window - Window size
        :return scores - List of the final scores of the individual exam simulations """


    rolling_mean, rolling_std, momentum, z_score = [], [], [], []

    if additional_stats:

        # Rolling mean
        rolling_mean = df['score_percentage'].rolling(window, min_periods=1).mean()

        # Rolling std
        rolling_std = df['score_percentage'].rolling(window, min_periods=1).std()
        rolling_std = rolling_std.fillna(0)

        # Momentum
        momentum = df['score_percentage'] - df['score_percentage'].shift(1)
        momentum = momentum.fillna(0)

        # Z-score
        z_score = (df['score_percentage'] - rolling_mean) / rolling_std
        z_score = z_score.fillna(0)

    student_scores = {
                      'scores': list(df['score_percentage']),
                      'rolling_mean': list(rolling_mean),
                      'rolling_std': list(rolling_std),
                      'momentum': list(momentum),
                      'z_score': list(z_score)
                      }


    return student_scores

def calculate_confidence_interval_t_distribution(m: float,
                                    std: float,
                                    n: int,
                                    df: int,
                                    confidence_level: float = 0.95):
        """ Standard Error based Confidence Interval Estimation
            :param m - Mean value
            :param std - Standard Deviation
            :param n - Sample size
            :param df - Degrees of freedom
            :param confidence_level - Desired confidence level
            :return confidence_interval """

        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df)

        return [m - t_critical*(std/np.sqrt(n)), m + t_critical*(std/np.sqrt(n))]
