import numpy as np

class Measurement:
    def __init__(self,x,y,sx,sy):
        self.xy = np.array([x,y])
        self.sxy = np.array([sx,sy]) 