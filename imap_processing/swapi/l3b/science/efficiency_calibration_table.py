class EfficiencyCalibrationTable:
    def __init__(self, path):
        self.index = 0

    def get_efficiency_for(self, time):
        self.index += 1
        if self.index < 3:
            return 0.1
        if self.index < 5:
            return 0.09
        return 0.0882
