"""
Extract data about a replay into a csv.
"""

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys
import os

class MainClient(Client):
    def __init__(self) -> None:
        cntExistingFiles = len(next(os.walk("."))[2])
        self.fout = open(str(cntExistingFiles) + ".csv", "w")

        self.state = None
        self.finished = False
        self.race_time = 0
        super(MainClient, self).__init__()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_simulation_begin(self, iface: TMInterface):
        self.fout.write("time, x, y, z, yaw, pitch, roll, vx, vy, vz, " +
                   "wheel0_material, wheel1_material, wheel2_material, wheel3_material, " +
                   "wheel0_has_contact, wheel1_has_contact, wheel2_has_contact, wheel3_has_contact, " +
                   "wheel0_is_sliding, wheel1_is_sliding, wheel2_is_sliding, wheel3_is_sliding, " +
                   "gas, brake, steer\n")

        iface.remove_state_validation()
        self.finished = False

    def on_simulation_step(self, iface: TMInterface, _time: int):
        if self.finished:
            return

        self.race_time = _time
        if self.race_time >= 0:
            self.state = iface.get_simulation_state()

        if self.state:
            x, y, z = [round(a, 5) for a in self.state.position]
            yaw, pitch, roll = [round(a, 5) for a in self.state.yaw_pitch_roll]
            vx, vy, vz = [round(a, 5) for a in self.state.velocity]
            #TODO wheel?_angle.
            wheel_materials = [self.state.simulation_wheels[i].real_time_state.contact_material_id for i in range(4)]
            wheel_has_contact = [1 if self.state.simulation_wheels[i].real_time_state.has_ground_contact else 0 for i in range(4)]
            wheel_is_sliding = [1 if self.state.simulation_wheels[i].real_time_state.is_sliding else 0 for i in range(4)]
            gas = 1 if self.state.input_accelerate else 0
            brake = 1 if self.state.input_brake else 0
            steer = self.state.input_steer
            if self.state.input_left or self.state.input_right:
                steer = 0 if (self.state.input_left and self.state.input_right) else (-65536 if self.state.input_left else 65536)

            self.fout.write(f"{self.race_time}, {x}, {y}, {z}, {yaw}, {pitch}, {roll}, {vx}, {vy}, {vz}, " +
                        ''.join([str(x) + ", " for x in wheel_materials]) +
                        ''.join([str(x) + ", " for x in wheel_has_contact]) +
                        ''.join([str(x) + ", " for x in wheel_is_sliding]) +
                        f"{gas}, {brake}, {steer}\n")

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        print(f'Reached checkpoint {current}/{target}')
        if current == target:
            print(f'Finished the race at {self.race_time}')
            self.finished = True
            iface.prevent_simulation_finish()

    def on_simulation_end(self, iface: TMInterface, result: int):
        print('Simulation finished')
        self.fout.close()
        iface.close()


def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)


if __name__ == '__main__':
    main()