import sys
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from xml.dom.minidom import parse
from math import cos, asin, sqrt, pi


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.7f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.7f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


def get_data(input_gpx):
    doc = parse(input_gpx)
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

    data = pd.DataFrame(columns=['lat', 'lon', 'datetime'])
    data['lat'] = lat_list
    data['lon'] = lon_list
    data['datetime'] = datetime_list
    # print(data)
    return data


# inner helper function is adapted from https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def distance(points):
    points['lat2'] = points['lat'].shift(1)
    points['lon2'] = points['lon'].shift(1)
    
    def inner_func(lat1, lon1, lat2, lon2):
        p = pi/180
        a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
        return 12742 * asin(sqrt(a)) #2*R*asin...
    
    distance = points.apply(lambda x: inner_func(x['lat'], x['lon'], x['lat2'], x['lon2']), axis=1)
    points.drop(labels=['lat2', 'lon2'], axis=1, inplace=True)
    return distance.sum() * 1000



def smooth(points):
    kalman_data = points[['lat', 'lon', 'Bx', 'By']]
    observation_covariance = np.diag([5e-5, 5e-5, 5e-5, 5e-5]) ** 2
    initial_state_mean = kalman_data.iloc[0]
    transition_matrices = [[1, 0, 5e-7, 34e-7], [0, 1, -49e-7, 9e-7], [0, 0, 1, 0], [0, 0, 0, 1]]
    transition_covariance = np.diag([5e-5, 5e-5, 5e-5, 5e-5]) ** 2

    kf = KalmanFilter(
        initial_state_mean=initial_state_mean, 
        observation_covariance=observation_covariance, 
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrices)
    
    kalman_smoothed, _ = kf.smooth(kalman_data)
    data = pd.DataFrame(kalman_smoothed, columns=['lat', 'lon', 'Bx', 'By'])
    # print(data)
    return data
    


def main():
    input_gpx = sys.argv[1]
    input_csv = sys.argv[2]
    
    points = get_data(input_gpx).set_index('datetime')
    # print(points)
    sensor_data = pd.read_csv(input_csv, parse_dates=['datetime']).set_index('datetime')
    points['Bx'] = sensor_data['Bx']
    points['By'] = sensor_data['By']

    dist = distance(points)
    print(f'Unfiltered distance: {dist:.2f}')

    smoothed_points = smooth(points)
    smoothed_dist = distance(smoothed_points)
    print(f'Filtered distance: {smoothed_dist:.2f}')

    output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()
