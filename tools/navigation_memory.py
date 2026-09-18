import numpy as np

class NavigationMemory:
    """Component 3: Trajectory Buffer Matrices (Short vs Long)"""
    def __init__(self, frame_trigger=90, dist_trigger=50.0):
        self.short_term_buffer = {
            'path': [],
            'anchors': []
        }
        self.long_term_buffer = {
            'path': [],
            'anchors': []
        }
        self.frame_trigger = frame_trigger
        self.dist_trigger = dist_trigger
        self.frames_since_checkpoint = 0
        self.last_checkpoint_pos = None

    def add_step(self, current_pos, active_anchors):
        if self.last_checkpoint_pos is None:
            self.last_checkpoint_pos = current_pos
            
        self.short_term_buffer['path'].append(current_pos.copy())
        self.short_term_buffer['anchors'].append(active_anchors)
        self.frames_since_checkpoint += 1
        
        dist_moved = np.linalg.norm(current_pos - self.last_checkpoint_pos)
        
        if self.frames_since_checkpoint >= self.frame_trigger or dist_moved >= self.dist_trigger:
            self.flush_checkpoint(current_pos)

    def flush_checkpoint(self, current_pos):
        self.long_term_buffer['path'].extend(self.short_term_buffer['path'])
        self.long_term_buffer['anchors'].extend(self.short_term_buffer['anchors'])
        
        self.short_term_buffer['path'] = []
        self.short_term_buffer['anchors'] = []
        self.frames_since_checkpoint = 0
        self.last_checkpoint_pos = current_pos
