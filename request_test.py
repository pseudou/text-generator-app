"""
    testing the api
"""
import requests

params={"initial_text": "the","length":40}

resp = requests.post("http://localhost:5000/generate/",params=params)

print(resp.json())