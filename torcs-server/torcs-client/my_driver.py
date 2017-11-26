from pytocl.driver import Driver
from pytocl.car import State, Command, KMH_PER_MPS
from models import net
import numpy as np

class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    ...
    def drive(self, carstate: State) -> Command:
        x = np.asarray([[carstate.speed_x, carstate.distance_from_center, carstate.angle,
             *carstate.distances_from_edge]])
        y = net.predict(x)

        command = Command()

        self.steer(carstate, y[2], command)

        if (carstate.speed_x * KMH_PER_MPS) < 80 and carstate.distance_raced < 100:
            self.accelerate(carstate, 100, command)
        else:
            self.accelerate(carstate, y[3], command)
            command.accelerator = y[0]
            command.brake = y[1]

        return command
