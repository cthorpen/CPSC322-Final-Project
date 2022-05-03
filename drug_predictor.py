import requests  # HTTP requests
import json  # parsing strings/JSON objects

url = "https://drug-effectiveness.herokuapp.com/predict?"
# url = "http://localhost:5001/predict?"
# query items
url += "Drug=benzonatate&Age=0-2&Condition=Cough&Season=summer&EaseofUse=E&Satisfaction=E&Sex=Female"

# make the GET request
# links to Mozilla documentation in Gina's notes
response = requests.get(url)
# first check the status code
print("status code:", response.status_code)
# more links to documentation
if response.status_code == 200:
    # OK
    # parse the message body JSON
    json_obj = json.loads(response.text)
    print(type(json_obj))
    print(json_obj)
