from vehicle import Vehicle


class Detection:
    position = None
    frame_detected = -1
    last_update = -1
    vehicle = Vehicle.none

    def __init__(self):
        self.counted = False
