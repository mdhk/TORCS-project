from pytocl.driver import Driver
from pytocl.car import State, Command, KMH_PER_MPS

# network = "models/mlp101"
# net = nn.Module.load_state_dict(torch.load(network))

class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    ...
    def drive(self, carstate: State) -> Command:

        command = Command()

        away = False
        for state in carstate.distances_from_edge:
            if state < 1:
                away = True
        if 1==2:
            print("hoi")
        else:
            # command = Command()

            self.steer(carstate, 0.0, command)

            self.accelerate(carstate, 10000, command)

            # log data
            if self.data_logger:
                self.data_logger.log(carstate, command)

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
