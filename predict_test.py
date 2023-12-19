import requests

url = 'http://localhost:9696/predict'

individual = {
'gender': 'male',
'race_ethnicity': 'group E',
'parental_level_of_education': 'some college',
'lunch': 'standard',
'test_preparation_course': 'completed',
'total score': 245,
'average': 81.66666666666667
}

response = requests.post(url, json=individual).json()

print(response)


