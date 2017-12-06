from pytocl.driver import Driver
from pytocl.car import State, Command, KMH_PER_MPS
import numpy as np
import datetime
import keyboard

class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    ...
    def __init__(self, logdata=False):
        self.last_steer = 0.
        date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%m-%S')
        filename = './drivelogs/drivelog-HUMAN-_{}.csv'.format(date_time)
        self.drivelog = open(filename, 'w', 10)
        columns = ['ACCELERATOR', 'BRAKE', 'GEAR', 'STEERING', 'ANGLE', \
        'CURRENT_LAP_TIME', 'DAMAGE', 'DISTANCE_FROM_START', \
        'DISTANCE_RACED', 'LAST_LAP_TIME', 'OPPONENT_1', 'OPPONENT_2', \
        'OPPONENT_3', 'OPPONENT_4', 'OPPONENT_5', 'OPPONENT_6', 'OPPONENT_7', \
        'OPPONENT_8', 'OPPONENT_9', 'OPPONENT_10', 'OPPONENT_11', 'OPPONENT_12', \
        'OPPONENT_13', 'OPPONENT_14', 'OPPONENT_15', 'OPPONENT_16', 'OPPONENT_18', \
        'OPPONENT_19', 'OPPONENT_20', 'OPPONENT_21', 'OPPONENT_22', 'OPPONENT_23', \
        'OPPONENT_24', 'OPPONENT_25', 'OPPONENT_26', 'OPPONENT_27', 'OPPONENT_28', \
        'OPPONENT_29', 'OPPONENT_30', 'OPPONENT_31', 'OPPONENT_32', 'OPPONENT_33', \
        'OPPONENT_34', 'OPPONENT_35', 'OPPONENT_36', 'RACE_POSITION', \
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

    def drive(self, carstate: State) -> Command:
        accelerator, brake, steering = (0., 0., 0.)
        gear = carstate.gear
        # print(carstate.sensor_dict)

        command = Command()

        if keyboard.is_pressed('up'):
            accelerator += 10000
        if keyboard.is_pressed('down'):
            brake += 1
        if keyboard.is_pressed('left'):
            steering = self.last_steer + 0.04
        if keyboard.is_pressed('right'):
            steering = self.last_steer - 0.04

        if keyboard.is_pressed('1'):
            gear = 1
        if keyboard.is_pressed('2'):
            gear = 2
        if keyboard.is_pressed('3'):
            gear = 3
        if keyboard.is_pressed('4'):
            gear = 4
        if keyboard.is_pressed('5'):
            gear = 5
        if keyboard.is_pressed('6'):
            gear = 6
        if keyboard.is_pressed('q'):
            gear = carstate.gear - 1
        if keyboard.is_pressed('w'):
            gear += carstate.gear + 1

        self.last_steer = steering

        if carstate.rpm > 9000:
            gear = carstate.gear + 1
        elif carstate.rpm < 2500 and gear >= 0:
            gear = carstate.gear - 1

        # if not carstate.gear in [-1, 1, 2,3,4,5,6]:
        #     gear = 1

        command.accelerator = accelerator
        command.brake = brake
        command.steering = steering
        command.gear = gear

        if not command.gear:
            command.gear = carstate.gear or 1
        # if command.gear < 1:
        #     command.gear = 1
        if carstate.speed_x <= 0:
            if keyboard.is_pressed('down'):
                command.gear = -1
                command.brake = 0
                command.accelerator +=1
            elif keyboard.is_pressed('up') and carstate.speed_x < -1:
                command.gear = 1
                command.brake = 1
                command.accelerator +=1

        drivelog_data = [command.accelerator, command.brake, command.gear, \
        command.steering, carstate.angle, carstate.current_lap_time, \
        carstate.damage, carstate.distance_from_start, carstate.distance_raced,
        carstate.last_lap_time, *carstate.opponents, carstate.race_position, \
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
