from vehicle import Vehicle


class Detection:
    position = None
    frame_detected = -1
    last_update = -1
    vehicle = Vehicle.none

    def __init__(self):
        self.counted = False

    def copy(self):
        result = Detection()

        result.counted = self.counted
        result.position = self.position
        result.frame_detected = self.frame_detected
        result.last_update = self.last_update
        result.vehicle = self.vehicle

        return result
