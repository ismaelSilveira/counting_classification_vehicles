from vehicle_counting import Vehicle


class Detection:
    position = None
    size = (0, 0)
    frame_detected = -1
    last_update = -1
    vehicle = Vehicle.none

    def __init__(self):
        self.counted = False
