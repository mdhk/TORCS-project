from pytocl.driver import Driver
from pytocl.car import State, Command, KMH_PER_MPS
from mlp import net
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

        command.accelerator = y[0]

        ### Uncomment to disable brakes for more fun
        # if np.abs(y[1]) <= 0.03 or (carstate.speed_x * KMH_PER_MPS) < 100 or \
        # carstate.distance_raced < 80:
        #     command.brake = 0
        # else:
        #     command.brake = y[1]
        #
        # self.accelerate(carstate, y[3], command)
        ###

        ### Uncomment to start fast, then drive like grandma
        # if (carstate.speed_x * KMH_PER_MPS) < 100 and carstate.distance_raced < 50:
        #     self.accelerate(carstate, 100, command)
        # else:
        #     self.accelerate(carstate, y[3], command)
        #     command.brake = y[1]
        ###

        ### Uncomment to use only MLP predictions (be like grandma always)
        self.accelerate(carstate, y[3], command)
        command.brake = y[1]
        ###

        return command
