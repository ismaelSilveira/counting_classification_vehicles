class Detection:
    vehicle = None
    frame_detected = -1
    last_update = -1

    def __init__(self):
        self.counted = False
