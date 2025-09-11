import pandas as pd
import urllib3
import warnings

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

from helpers import calculate_ci_percentage_above_threshold, load_single_student_simluation_scores, calculate_confidence_interval_t_distribution

urllib3.disable_warnings()
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def arima(df: pd.DataFrame, WINDOW_SIZE = 5, CI_method = 't_distribution'):    
    student_dict = load_single_student_simluation_scores(df=df,
                                                        additional_stats=True,
                                                        window=WINDOW_SIZE)
    
    try:            
            scores = student_dict['scores'].copy()  
            rolling_std = student_dict['rolling_std']

            # ********** Predictions and Confidence Intervals **********

            # First do a search to find the best ARIMA(p,d,q) order)
            best_order = auto_arima(
                scores,
                seasonal=False,  # No seasonality (ARIMA, not SARIMA)
                trace=False, # Don’t print the progress/logging
                error_action='ignore', # Ignore models that fail to fit
                suppress_warnings=True, # Suppress warnings (e.g., convergence issues)
                stepwise=True # Use stepwise algorithm (faster than full grid search)
            ).order
            
            # If Stepwise algorithm fails, use Grid-Search
            if best_order == (0, 0, 0):
                best_order = auto_arima(
                    scores,
                    seasonal=False,  # No seasonality (ARIMA, not SARIMA)
                    trace=False,  # Don’t print the progress/logging
                    error_action='ignore',  # Ignore models that fail to fit
                    suppress_warnings=True,  # Suppress warnings (e.g., convergence issues)
                    stepwise=False  # Use full grid search algorithm (slower than stepwise)
                ).order

            # If Grid-Search fails, use a fallback option
            if best_order == (0, 0, 0): 
                best_order = (2, 1, 2)

            # ARIMA fit and forecast
            model = ARIMA(scores, order=best_order)
            model_fit = model.fit()

            forecast_result = model_fit.get_forecast(steps=1)
            predicted_value = forecast_result.predicted_mean[0]

            # With alpha=0.05 we get confidence interval for 95% confidence level
            conf_int = forecast_result.conf_int(alpha=0.05)
            lower_bound = conf_int[0, 0]
            upper_bound = conf_int[0, 1]
            CI_arima = [lower_bound, upper_bound]

            #Normal distribution Confidence Interval
            alpha = 2
            rolling_lower_bound = predicted_value - alpha * rolling_std[-1]
            rolling_upper_bound = predicted_value + alpha * rolling_std[-1]
            CI_normal_distribution = [rolling_lower_bound, rolling_upper_bound]

            CI_t_distribution = calculate_confidence_interval_t_distribution(predicted_value,
                                                    rolling_std[-1],
                                                    WINDOW_SIZE,
                                                    WINDOW_SIZE - 1)

            CI_arima_above_threshold_pct = calculate_ci_percentage_above_threshold(CI_arima)
            CI_normal_distribution_above_threshold_pct = calculate_ci_percentage_above_threshold(CI_normal_distribution)
            CI_t_distribution_above_threshold_pct = calculate_ci_percentage_above_threshold(CI_normal_distribution)
            
            if CI_method == "t_distribution":
                 CI = CI_t_distribution
                 CI_above_threshold_pct = CI_t_distribution_above_threshold_pct
            elif CI_method == "normal_distribution":
                 CI = CI_normal_distribution
                 CI_above_threshold_pct = CI_normal_distribution_above_threshold_pct
            elif CI_method == "arima_distribution":
                 CI = CI_arima
                 CI_above_threshold_pct = CI_arima_above_threshold_pct
            else:
                 CI = "CI_method_not_found"
                 CI_above_threshold_pct = "CI_above_threshold_pct_method_not_found"

            result = {
                "probability_of_passing_final_exam": CI_above_threshold_pct,
                "predicted_simulation_score": predicted_value,
                "arima_best_order": best_order,
                "CI_from_arima": CI
            }
            return result
            
    except Exception as e:
        print(f"Error processing student scores: {str(e)}")
        return {
            "error": str(e),
            "predictions": None,
            "confidence_intervals": None
        }
