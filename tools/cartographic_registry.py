import numpy as np

class CartographicRegistry:
    """Component 2: Global Map Registry & Spatial NMS"""
    def __init__(self, merge_threshold=7.5):
        """
        :param merge_threshold: Distance in meters to merge objects
        """
        self.merge_threshold = merge_threshold
        self.registered_objects = {}  # {id: {'pos_relative': np.array, 'pos_pixel': np.array, 'cls': int, 'last_seen': int}}
        self.next_id = 1
        
    def register_or_update(self, new_centroids_pixel, classes, toolkit_instance, frame_count):
        """
        new_centroids_pixel: list of (px, py)
        classes: list of class IDs corresponding to new_centroids_pixel
        toolkit_instance: CoordinateToolkit instance
        """
        active_ids = []
        
        for (px, py), cls_id in zip(new_centroids_pixel, classes):
            rel_pos = toolkit_instance.pixel_to_relative(px, py)
            
            merged = False
            for obj_id, obj_data in self.registered_objects.items():
                if obj_data['cls'] != cls_id:
                    continue
                
                # Check spatial distance using relative coordinates (meters)
                dist = np.linalg.norm(obj_data['pos_relative'] - rel_pos)
                if dist < self.merge_threshold:
                    # Update moving average
                    alpha = 0.5
                    self.registered_objects[obj_id]['pos_relative'] = (alpha * self.registered_objects[obj_id]['pos_relative']) + ((1 - alpha) * rel_pos)
                    self.registered_objects[obj_id]['pos_pixel'] = (alpha * self.registered_objects[obj_id]['pos_pixel']) + ((1 - alpha) * np.array([px, py]))
                    self.registered_objects[obj_id]['last_seen'] = frame_count
                    active_ids.append(obj_id)
                    merged = True
                    break
                    
            if not merged:
                # Register new object
                obj_id = self.next_id
                self.next_id += 1
                self.registered_objects[obj_id] = {
                    'pos_relative': rel_pos,
                    'pos_pixel': np.array([px, py]),
                    'cls': cls_id,
                    'last_seen': frame_count
                }
                active_ids.append(obj_id)
                
        # Return currently active anchors in the format expected by visualization and algorithm
        active_anchors = []
        for obj_id in active_ids:
            active_anchors.append({
                'id': obj_id,
                'pos_pixel': self.registered_objects[obj_id]['pos_pixel'],
                'pos_relative': self.registered_objects[obj_id]['pos_relative'],
                'cls': self.registered_objects[obj_id]['cls']
            })
            
        return active_anchors
