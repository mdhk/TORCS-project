from pytocl.driver import Driver
from pytocl.car import State, Command, KMH_PER_MPS
import torch
import networks as nw
import numpy as np
import torch.nn as nn
from train import net

# network = "models/mlp101"
# net = nn.Module.load_state_dict(torch.load(network))

class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    ...
    def drive(self, carstate: State) -> Command:
        """
        Wat is dit
        """
        x = np.asarray([[carstate.speed_x*KMH_PER_MPS, carstate.distance_from_center, carstate.angle,
             *carstate.distances_from_edge]])
        y = net.predict(x)

        command = Command()

        self.steer(carstate, y[2], command)

        command.accelerator = y[0]

        # ## Uncomment to disable brakes for more fun
        if np.abs(y[1]) <= 0.03 or (carstate.speed_x * KMH_PER_MPS) < 90 or \
        carstate.distance_raced < 100:
            command.brake = 0
            self.accelerate(carstate, 95, command)
        else:
            command.brake = y[1]
            self.accelerate(carstate, y[3], command)
        # ##

        # ## Uncomment to start fast, then drive like grandma
        # if (carstate.speed_x * KMH_PER_MPS) < 100 and carstate.distance_raced < 50:
        #     self.accelerate(carstate, 100, command)
        # else:
        #     self.accelerate(carstate, y[3], command)
        #     command.brake = y[1]
        # ##

        # ## Uncomment to use only MLP predictions (be like grandma always)
        # self.accelerate(carstate, y[3], command)
        # command.brake = y[1]
        # ##

        ## Uncomment to use only MLP predictions with small brake
        # self.accelerate(carstate, carstate.speed_x, command)
        # command.brake = y[1]
        # self.accelerate(carstate, y[3], command)
        # command.brake = y[1]/10
        ##

        # log data
        self.data_logger.log(carstate, command)

        return command
