from enum import Enum
from utils import calculate_cars_area


class Vehicle(Enum):
    bike = 1
    car = 2
    truck = 3
    none = 4


class VehicleCounting:
    count = 0
    bikes = []
    cars = []
    trucks = []
    cars_area = 0

    def __init__(self):
        pass

    def classify_vehicles(self, vehicles):
        self.cars_area = calculate_cars_area(vehicles)
