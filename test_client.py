# client.py
import requests

payload = {
    "scores": [0.9, 0.88, 0.75, 0.78, 0.74, 0.69],
    "window_size": 5,
    "ci_method": "t_distribution"
}

response = requests.post("http://127.0.0.1:8000/forecast", json=payload)

if response.status_code == 200:
    print("Forecast:", response.json())
else:
    print("Error:", response.status_code, response.text)
