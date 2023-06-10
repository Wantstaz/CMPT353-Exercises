import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# adapted from my exer3 distance()
# inner helper function is adapted from https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def distance(city, stations):
    lat1 = city['latitude']
    lon1 = city['longitude']
    lat2 = stations['latitude']
    lon2 = stations['longitude']

    p = np.pi / 180
    a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a))


def best_tmax(city, stations):
    distances = distance(city, stations)
    min_distance_idx = distances.idxmin()
    return stations.loc[min_distance_idx, 'avg_tmax']


def main():
    stations_file = sys.argv[1]
    city_file = sys.argv[2]
    output_file = sys.argv[3]

    stations = pd.read_json(stations_file, lines=True)
    cities = pd.read_csv(city_file)
    cities = cities.dropna(subset=['area', 'population'])
    cities['area'] = cities['area'] / 1000000
    cities = cities[cities['area'] <= 10000]

    cities['avg_tmax'] = cities.apply(best_tmax, stations=stations, axis=1)
    cities['avg_tmax'] /= 10

    cities['density'] = cities['population'] / cities['area']

    plt.scatter(cities['avg_tmax'], cities['density'], alpha=0.5)
    plt.ylabel('Population Density (people/km²)')
    plt.xlabel('Avg Max Temperature (°C)')
    plt.title('Temperature vs Population Density')
    plt.savefig(output_file)
    # plt.show()


if __name__ == '__main__':
    main()
