import requests  # HTTP requests
import json  # parsing strings/JSON objects

# url = "https://name-of-our-app.herokuapp.com/predict?"
url = "http://127.0.0.1:5001/predict?"
# query items
url += "Drug=zyrtec-d&Age=19-24&Condition=Other&Season=summer&EaseofUse=5.0&Satisfaction=3.0&Sex=Female"

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