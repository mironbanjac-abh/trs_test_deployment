# client.py
import requests

payload = {
    "scores": [0.9, 0.88, 0.75, 0.78, 0.74, 0.69, 0.95, 0.88, 0.9, 0.63, 0.93, 0.95, 0.94, 0.56, 0.76, 0.99, 0.89, 0.99, 0.93, 0.99],
    "window_size": 10
}

response = requests.post("http://127.0.0.1:8000/forecast", json=payload)

if response.status_code == 200:
    print("Forecast:", response.json())
else:
    print("Error:", response.status_code, response.text)
