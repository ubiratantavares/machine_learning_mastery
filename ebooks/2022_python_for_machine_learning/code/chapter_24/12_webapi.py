import pandas as pd
import matplotlib.pyplot as plt
import requests


# Create query URL for list of countries, by default only 50 entries returned per page
url = "http://api.worldbank.org/v2/country/all?format=json&per_page=500"
response = requests.get(url)
# Expects HTTP status code 200 for correct query
print(response.status_code)
# Get the response in JSON
header, data = response.json()
print(header)
# Collect a list of 3-letter country code excluding aggregates
countries = [item["id"]
             for item in data
             if item["region"]["value"] != "Aggregates"]
print(countries)


# Create query URL for total population from all countries in 2020
arguments = {
    "country": "all",
    "indicator": "SP.POP.TOTL",
    "date": 2020,
    "format": "json"
}
url = "http://api.worldbank.org/v2/country/{country}/" \
      "indicator/{indicator}?date={date}&format={format}&per_page=500"
query_population = url.format(**arguments)
response = requests.get(query_population)
print(response.status_code)
# Get the response in JSON
header, population_data = response.json()
print(header)


# Filter for countries, not aggregates
population = []
for item in population_data:
    if item["countryiso3code"] in countries:
        name = item["country"]["value"]
        population.append({"country":name, "population": item["value"]})
# Create DataFrame for sorting and filtering
population = pd.DataFrame.from_dict(population)
population = population.dropna().sort_values("population").iloc[-25:]
# Plot bar chart
fig = plt.figure(figsize=(15,7))
plt.bar(population["country"], population["population"]/1e6)
plt.xticks(rotation=90)
plt.ylabel("Million Population")
plt.title("Population")
plt.show()
