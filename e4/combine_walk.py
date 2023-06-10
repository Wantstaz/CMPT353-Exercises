import os
import pathlib
import sys
import numpy as np
import pandas as pd

from xml.dom.minidom import parse
from math import cos, asin, sqrt, pi


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation, parse
    xmlns = 'http://www.topografix.com/GPX/1/0'

    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.10f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.10f' % (pt['lon']))
        time = doc.createElement('time')
        time.appendChild(doc.createTextNode(pt['datetime'].strftime("%Y-%m-%dT%H:%M:%SZ")))
        trkpt.appendChild(time)
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    doc.documentElement.setAttribute('xmlns', xmlns)

    with open(output_filename, 'w') as fh:
        fh.write(doc.toprettyxml(indent='  '))


# adapted from my exer3 get_data()
def get_data(input_gpx):
    doc = parse(open(input_gpx))
    root = doc.documentElement
    trkpt = root.getElementsByTagName('trkpt')
    datetime_list = []
    lat_list = []
    lon_list = []

    for ele in trkpt:
        lat = float(ele.getAttribute('lat'))
        lon = float(ele.getAttribute('lon'))
        time = ele.getElementsByTagName('time')[0].childNodes[0].data
        datetime_list.append(pd.to_datetime(time, utc=True))
        lat_list.append(lat)
        lon_list.append(lon)

    data = pd.DataFrame(columns=['lat', 'lon', 'timestamp'])
    data['lat'] = lat_list
    data['lon'] = lon_list
    data['timestamp'] = datetime_list

    return data


# adapted from my exer3 distance()
# inner helper function is adapted from https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def distance(points):
    points['lat2'] = points['lat'].shift(1)
    points['lon2'] = points['lon'].shift(1)

    def inner_func(row):
        lat1 = row['lat']
        lon1 = row['lon']
        lat2 = row['lat2']
        lon2 = row['lon2']

        p = pi / 180
        a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
        return 12742 * asin(sqrt(a))

    points['distance'] = points.apply(inner_func, axis=1)
    total_distance = points['distance'].sum() * 1000
    points.drop(labels=['lat2', 'lon2', 'distance'], axis=1, inplace=True)

    return total_distance


def main():
    input_directory = pathlib.Path(sys.argv[1])
    output_directory = pathlib.Path(sys.argv[2])

    accl = pd.read_json(input_directory / 'accl.ndjson.gz', lines=True, convert_dates=['timestamp'])[['timestamp', 'x']]
    gps = get_data(input_directory / 'gopro.gpx')
    phone = pd.read_csv(input_directory / 'phone.csv.gz')[['time', 'gFx', 'Bx', 'By']]

    first_time = accl['timestamp'].min()

    # # Combining the Data
    # offset = 0
    # first_time = accl['timestamp'].min()
    # phone['timestamp'] = first_time + pd.to_timedelta(phone['time'] + offset, unit='sec')
    #
    # accl['timestamp'] = accl['timestamp'].dt.round('4S')
    # gps['timestamp'] = gps['timestamp'].dt.round('4S')
    # phone['timestamp'] = phone['timestamp'].dt.round('4S')
    #
    # accl = accl.groupby(['timestamp']).mean().reset_index()
    # gps = gps.groupby(['timestamp']).mean().reset_index()
    # phone = phone.groupby(['timestamp']).mean().reset_index()

    # Correlating Data Sets
    accl['timestamp'] = accl['timestamp'].dt.round('4S')
    gps['timestamp'] = gps['timestamp'].dt.round('4S')

    accl = accl.groupby(['timestamp']).mean().reset_index()
    gps = gps.groupby(['timestamp']).mean().reset_index()

    gopro_data = pd.merge(accl, gps, on='timestamp')

    highest_cross_corr = 0
    best_offset = 0

    combined = pd.DataFrame()
    for offset in np.linspace(-5.0, 5.0, 101):
        phone['timestamp'] = first_time + pd.to_timedelta(phone['time'] + offset, unit='sec')
        phone_temp = phone.copy()  # deep copy to avoid 4 second bins modify data
        phone_temp['timestamp'] = phone_temp['timestamp'].dt.round('4S')
        phone_temp = phone_temp.groupby(['timestamp']).mean().reset_index()
        combined_temp = pd.merge(phone_temp, gopro_data, on='timestamp')

        cross_corr_temp = (combined_temp['gFx'] * combined_temp['x']).sum()
        if cross_corr_temp > highest_cross_corr:
            highest_cross_corr = cross_corr_temp
            best_offset = offset
            combined = combined_temp

    combined.rename(columns={'timestamp': 'datetime'}, inplace=True)

    print(f'Best time offset: {best_offset:.1f}')
    os.makedirs(output_directory, exist_ok=True)
    output_gpx(combined[['datetime', 'lat', 'lon']], output_directory / 'walk.gpx')
    combined[['datetime', 'Bx', 'By']].to_csv(output_directory / 'walk.csv', index=False)



main()
