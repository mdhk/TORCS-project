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
    def __init__(self, logdata=False):
        self.last_steer = 0.
        date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%m-%S')
        filename = './drivelogs/drivelog-_{}.csv'.format(date_time)
        self.drivelog = open(filename, 'w', 10)
        columns = ['ACCELERATOR', 'BRAKE', 'GEAR', 'STEERING', 'ANGLE', \
        'CURRENT_LAP_TIME', 'DAMAGE', 'DISTANCE_FROM_START', \
        'DISTANCE_RACED', 'LAST_LAP_TIME', 'OPPONENTS', 'RACE_POSITION', \
        'RPM', 'SPEED_X', 'SPEED_Y', 'SPEED_Z', 'DISTANCE_FROM_CENTER', \
        'CENTR_OF_MASS_DISTANCE', 'WHEEL_V_1', 'WHEEL_V_2', 'WHEEL_V_3', \
        'WHEEL_V_4', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', \
        'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', \
        'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', \
        'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', \
        'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', \
        'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']
        self.drivelog.write(','.join(columns))
        self.drivelog.write('\n')
        self.drivelog.flush()

    def drive(self, carstate: State):
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
        drivelog_data = [command.accelerator, command.brake, command.gear, \
        command.steering, carstate.angle, carstate.current_lap_time, \
        carstate.damage, carstate.distance_from_start, carstate.distance_raced,
        carstate.last_lap_time, carstate.opponents, carstate.race_position, \
        carstate.rpm, carstate.speed_x, carstate.speed_y, carstate.speed_z, \
        carstate.distance_from_center, carstate.z, *carstate.wheel_velocities, \
        *carstate.distances_from_edge]

        if self.drivelog:
            self.drivelog.write(','.join([str(value) for value in drivelog_data]))
            self.drivelog.write('\n')

        return command

    def on_shutdown(self):
        """
        Server requested driver shutdown.

        Optionally implement this event handler to clean up or write data
        before the application is stopped.
        """
        if self.drivelog:
            self.drivelog.close()
            self.drivelog = None
