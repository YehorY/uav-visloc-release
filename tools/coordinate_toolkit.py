import numpy as np
import math

class CoordinateToolkit:
    """Component 1: Metric Translation Toolkit"""
    def __init__(self, gsd=0.5):
        """
        :param gsd: Ground Sample Distance (meters per pixel)
        """
        self.gsd = gsd
        
    def pixel_to_relative(self, px, py):
        """
        Translates bounding box centroids into relative spatial coordinates.
        Assuming the center of the image is (0,0) in relative coordinates, but 
        for distance calculation, direct scaling is sufficient.
        """
        return np.array([px * self.gsd, py * self.gsd])
        
    def relative_to_gps(self, dx, dy, base_lat, base_lon):
        """
        Calculates new GPS coordinates given a spatial delta from a base point.
        dx, dy are in meters.
        Approximation: 1 degree latitude ~ 111,320 meters.
        1 degree longitude ~ 111,320 * cos(latitude) meters.
        """
        lat_offset = dy / 111320.0
        lon_offset = dx / (111320.0 * math.cos(math.radians(base_lat)))
        return base_lat + lat_offset, base_lon + lon_offset
        
    def gps_to_radar_pixel(self, lat, lon, min_lat, max_lat, min_lon, max_lon, canvas_size=600):
        """Converts GPS to 2D canvas pixels for the Radar visualization."""
        # Add tiny padding
        lat_pad = (max_lat - min_lat) * 0.1 if max_lat > min_lat else 0.001
        lon_pad = (max_lon - min_lon) * 0.1 if max_lon > min_lon else 0.001
        
        padded_min_lat = min_lat - lat_pad
        padded_max_lat = max_lat + lat_pad
        padded_min_lon = min_lon - lon_pad
        padded_max_lon = max_lon + lon_pad
        
        # X-axis is Longitude (West to East)
        x = int((lon - padded_min_lon) / (padded_max_lon - padded_min_lon) * canvas_size)
        # Y-axis is Latitude (South to North), inverted for image (top-left is 0,0)
        y = int(canvas_size - (lat - padded_min_lat) / (padded_max_lat - padded_min_lat) * canvas_size)
        
        return x, y

    def gps_to_map_pixel(self, lat, lon, min_lat, max_lat, min_lon, max_lon, map_w, map_h):
        """Converts GPS to 2D canvas pixels for arbitrary map dimensions."""
        lat_pad = (max_lat - min_lat) * 0.1 if max_lat > min_lat else 0.001
        lon_pad = (max_lon - min_lon) * 0.1 if max_lon > min_lon else 0.001
        
        padded_min_lat = min_lat - lat_pad
        padded_max_lat = max_lat + lat_pad
        padded_min_lon = min_lon - lon_pad
        padded_max_lon = max_lon + lon_pad
        
        x = int((lon - padded_min_lon) / (padded_max_lon - padded_min_lon) * map_w)
        y = int(map_h - (lat - padded_min_lat) / (padded_max_lat - padded_min_lat) * map_h)
        
        return x, y

        
    def normalize_graph(self, anchors_list):
        """
        Normalizes the distances between a set of local anchors so they can be fed 
        into the EnsembleNavigator regardless of absolute pixel scale.
        """
        if not anchors_list:
            return []
        
        # Simple normalization: center them around their mean and scale max distance to 1
        anchors_array = np.array(anchors_list)
        mean_pos = np.mean(anchors_array, axis=0)
        centered = anchors_array - mean_pos
        
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        if max_dist > 0:
            normalized = centered / max_dist
        else:
            normalized = centered
            
        return normalized.tolist()
