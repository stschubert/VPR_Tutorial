from math import radians, cos, sin, asin, sqrt, pi

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6372.8  # For Earth radius in kilometers use 6372.8 km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))

    return R * c

# def distance(lat1, lon1, lat2, lon2):
#     r = 6371 # km
#     p = pi / 180

#     a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
#     return 2 * r * asin(sqrt(a))