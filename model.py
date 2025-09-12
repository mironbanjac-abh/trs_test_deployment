import pandas as pd
import urllib3
import warnings
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from helpers import calculate_ci_percentage_above_threshold, load_single_student_simulation_scores, calculate_confidence_interval_t_distribution

urllib3.disable_warnings()
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def arima(df: pd.DataFrame, window_size: int = 5, ci_method: str = 't_distribution') -> dict:
    student_dict = load_single_student_simulation_scores(df=df, additional_stats=True, window=window_size)

    try:
        scores = student_dict['scores'].copy()
        rolling_std = student_dict['rolling_std']

        # ********** Predictions and Confidence Intervals **********
        n = len(scores)

        # Edge case: if len(scores) <= 3 -> Then use only AR(1)
        if n <= 3:
            best_order = (1, 0, 0)
        else:
        # First do a search to find the best ARIMA(p,d,q) order)
            try:
                best_order = auto_arima(
                    scores,
                    seasonal=False,  # No seasonality (ARIMA, not SARIMA)
                    trace=False,  # Don’t print the progress/logging
                    error_action='ignore',  # Ignore models that fail to fit
                    suppress_warnings=True,  # Suppress warnings (e.g., convergence issues)
                    stepwise=True  # Use stepwise algorithm (faster than full grid search)
                ).order
            except IndexError:
                best_order = (1, 0, 0)

            # Grid search fallback if trivial
            if best_order == (0, 0, 0):
                try:
                    best_order = auto_arima(
                        scores,
                        seasonal=False,  # No seasonality (ARIMA, not SARIMA)
                        trace=False,  # Don’t print the progress/logging
                        error_action='ignore',  # Ignore models that fail to fit
                        suppress_warnings=True,  # Suppress warnings (e.g., convergence issues)
                        stepwise=False  # Use full grid search algorithm (slower than stepwise)
                    ).order
                except IndexError:
                    best_order = (1, 0, 0)

            # If Grid-Search fails, use a fallback option for n >= 4
            if best_order == (0, 0, 0):
                best_order = (2, 1, 2)

        # ARIMA fit and forecast
        model = ARIMA(scores, order=best_order)
        model_fit = model.fit()

        forecast_result = model_fit.get_forecast(steps=1)
        predicted_value = forecast_result.predicted_mean[0]

        # Calculating confidence interval based on request parameter
        ci = [0.0, 0.0]
        if ci_method == "t_distribution":
            ci = calculate_confidence_interval_t_distribution(predicted_value,
                                                              rolling_std[-1],
                                                              window_size,
                                                              window_size - 1)

        elif ci_method == "normal_distribution":
            alpha = 2
            rolling_lower_bound = predicted_value - alpha * rolling_std[-1]
            rolling_upper_bound = predicted_value + alpha * rolling_std[-1]
            ci = [rolling_lower_bound, rolling_upper_bound]

        elif ci_method == "arima_distribution":
            conf_int = forecast_result.conf_int(alpha=0.05)
            lower_bound = conf_int[0, 0]
            upper_bound = conf_int[0, 1]
            ci = [lower_bound, upper_bound]

        # Calculate probability of passing the final exam based on the confidence interval
        ci_above_threshold_pct = calculate_ci_percentage_above_threshold(ci)

        result = {
            "probability_of_passing_final_exam": ci_above_threshold_pct,
            "predicted_simulation_score": predicted_value,
            "arima_best_order": best_order,
            "confidence_interval_range": ci
            }

        return result
            
    except Exception as e:
        print(f"Error processing student scores: {str(e)}")
        return {
            "error": str(e),
            "predictions": None,
            "confidence_intervals": None
        }
