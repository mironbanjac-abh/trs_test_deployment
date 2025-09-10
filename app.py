from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

from model import arima

app = FastAPI()

class ForecastRequest(BaseModel):
    scores: List[float]
    window_size: int = 5

@app.post("/forecast")
def forecast_arima(request: ForecastRequest):
    # Validate scores are between 0.0 and 1.0
    for i, score in enumerate(request.scores):
        if not 0.0 <= score <= 1.0:
            raise HTTPException(
                status_code=400, 
                detail=f"Score at index {i} ({score}) must be between 0.0 and 1.0"
            )
    
    # Validate window_size is smaller than the number of scores
    if request.window_size >= len(request.scores):
        raise HTTPException(
            status_code=400,
            detail=f"window_size ({request.window_size}) must be smaller than the number of scores ({len(request.scores)})"
        )
    
    # Validate window_size is positive
    if request.window_size <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"window_size must be a positive integer"
        )
    
    try:
        df = pd.DataFrame(request.scores, columns=['score_percentage'])
        result = arima(df, WINDOW_SIZE=request.window_size)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))