class Detection:
    def __init__(self, vehicle, frame_number):
        self.vehicle = vehicle
        self.frame_detected = frame_number
        self.counted = False
