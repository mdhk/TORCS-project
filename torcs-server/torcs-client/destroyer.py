from pytocl.driver import Driver
from pytocl.car import State, Command, KMH_PER_MPS

# network = "models/mlp101"
# net = nn.Module.load_state_dict(torch.load(network))

class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    ...
    def drive(self, carstate: State) -> Command:
        if (1==2):
           print("haha")

        command = Command()
        a = 0
        self.accelerate(carstate, 10000, command)
        if carstate.speed_x < 10 and abs(carstate.angle) > 10:
            print("ik", carstate.distance_from_center, carstate.angle)
            if carstate.distance_from_center < 0 and carstate.angle > 0:
                command.gear = -1
                a = 1
            if carstate.distance_from_center > 0 and carstate.angle < 0:
                command.gear = -1
                a = 1
            # command.accelerator = 1000
            # self.steer(carstate, 0.0, command


        if a == 0:
            print("jij")
            # command = Command()


            if carstate.gear <= 0:
                carstate.gear = 1
            # self.accelerate(carstate, 10000, command)

            # log data
            if self.data_logger:
                self.data_logger.log(carstate, command)


        self.steer(carstate, 0.0, command)
        return command

    # def steer(self, carstate, target_track_pos, command):
    #     """
    #     codebode
    #     """
    #     steering_error = target_track_pos - carstate.distance_from_center
    #     steering_error -= carstate.speed_x /300 * steering_error
    #     command.steering = self.steering_ctrl.control(
    #         steering_error,
    #         carstate.current_lap_time
    #     )

    # @property
    # def range_finder_angles(self):
    #     """Iterable of 19 fixed range finder directions [deg].
    #
    #     The values are used once at startup of the client to set the directions
    #     of range finders. During regular execution, a 19-valued vector of track
    #     distances in these directions is returned in ``state.State.tracks``.
    #     """
    #     return -80, -60, -50, -40, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, \
    #         30, 40, 50, 60, 80
