from pytocl.driver import Driver
from pytocl.car import State, Command
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

        command.accelerator = y[0]
        command.brake = y[1]
        command.speed = y[2]

        if carstate.rpm > 7500:
            command.gear = carstate.gear + 1

        if carstate.rpm < 2500 and carstate.gear > 0:
            command.gear = carstate.gear - 1

        if carstate.gear < 1:
            carstate.gear = 1

        if not command.gear:
            command.gear = carstate.gear

        return command
