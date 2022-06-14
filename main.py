import numpy as np
import math
import matplotlib.pyplot as plt

def clamp(value, lower, upper, msg=""):
    if value < lower: print(msg); return lower
    if value > upper: print(msg); return upper
    return value

class PID():
    def __init__(self, K_p, K_i, K_d, target=0, min=-math.inf, max=math.inf):
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.target = target
        self.error_sum = 0
        self.min = min
        self.max = max
        
    def run(self, input, time_step):
        if not hasattr(self, 'last_error'): self.last_error = 0
        error = self.target - input
        error_rate = (error - self.last_error) / time_step
        self.error_sum += error * time_step
        output = self.K_p * error + \
                 self.K_i * self.error_sum + \
                 self.K_d * error_rate 
        self.last_error = error
        return clamp(output, self.min, self.max)

class TVCSimulation():
    # mass [kg]
    # moment_of_inertia [kg * m^2]
    # center_of_mass_to_tvc [m]
    # thrust [N]
    # time_step [s]
    # start_time [s]
    # stop_time [s]
    # gravity [m / s^2]
    # position_x [m]
    # position_z [m]
    # angular_position [rad]
    # velocity_x [m / s]
    # velocity_z [m / s]
    # angular_velocity [rad / s]
    # acceleration_x [m / s^2]
    # acceleration_z [m / s^2]
    # angular_acceleration [rad / s^2]
    # angle_limit [rad]
    # rate_limit [rad / s]
    def __init__(self,
                 mass, moment_of_inertia, center_of_mass_to_tvc, thrust,
                 time_step, start_time, stop_time, gravity,
                 position_x, position_z, angular_position,
                 velocity_x, velocity_z, angular_velocity,
                 acceleration_x, acceleration_z, angular_acceleration,
                 angle_limit, rate_limit,
                 K_p, K_i, K_d, pid_target, pid_min=-math.inf, pid_max=math.inf):
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.center_of_mass_to_tvc = center_of_mass_to_tvc
        self.thrust = thrust
        self.time_step = time_step
        assert stop_time > start_time
        self.start_time = start_time
        self.stop_time = stop_time
        self.steps = int((self.stop_time - self.start_time) / self.time_step) + 1
        self.gravity = gravity
        self.angle_limit = angle_limit
        self.step_rate_limit = rate_limit * self.time_step
        self.pid = PID(K_p, K_i, K_d, pid_target, pid_min, pid_max)
        self.position_x = position_x
        self.position_z = position_z
        self.angular_position = angular_position
        self.velocity_x = velocity_x
        self.velocity_z = velocity_z
        self.angular_velocity = angular_velocity
        self.acceleration_x = acceleration_x
        self.acceleration_z = acceleration_z
        self.angular_acceleration = angular_acceleration
    
    def loop(self):
        control_angle = 0
        positions_x = []
        positions_z = []
        angular_positions = []
        times = np.linspace(start=self.start_time, stop=self.stop_time, num=self.steps)
        for i in range(self.steps):
            force_z, force_x, moment = self.get_physics(control_angle)
            self.apply_physics(force_z, force_x, moment)
            positions_x.append(self.position_x)
            positions_z.append(self.position_z)
            angular_positions.append(np.rad2deg(self.angular_position))
            control_angle = self.pid.run(self.angular_position, self.time_step)
        self.graph(times, positions_x, positions_z, angular_positions)
            
    def get_physics(self, control_angle):
        if not hasattr(self, 'last_angle'): self.last_angle = control_angle        
        # Angle limitation
        control_angle = clamp(control_angle, -self.angle_limit, self.angle_limit, "WARNING: Clamping angle, outside of angle limit")
        control_angle = clamp(control_angle, self.last_angle - self.step_rate_limit, self.last_angle + self.step_rate_limit, "WARNING: Clamping angle, outside of angular rate limit")
        # thrust and moment calculations
        thrust_z = np.sin(control_angle) * self.thrust
        thrust_x = np.cos(control_angle) * self.thrust
        force_z = np.sin(self.angular_position) * thrust_z
        force_x = np.cos(self.angular_position) * thrust_x
        moment = thrust_z * self.center_of_mass_to_tvc
        self.last_angle = control_angle
        return force_z, force_x, moment
    
    def apply_physics(self, force_x, force_z, moment):
        # Obtain accelerations in each dimension
        acceleration_x = force_x / self.mass + self.gravity
        acceleration_z = force_z / self.mass
        angular_acceleration = moment / self.moment_of_inertia
        # Integrate acceleration in each dimension
        self.velocity_x += acceleration_x * self.time_step
        self.velocity_z += acceleration_z * self.time_step
        self.angular_velocity += angular_acceleration * self.time_step
        # Integrate velocity in each dimension
        self.position_x += self.velocity_x * self.time_step
        self.position_z += self.velocity_z * self.time_step
        self.angular_position += self.angular_velocity * self.time_step

    def graph(self, t, x, z, theta):
        figure, axis = plt.subplots(1, 3)
        print(t)
        print(x)
        print(z)
        print(theta)
        axis[0].plot(t, x)
        axis[0].set_title("X-position vs Time")
        axis[1].plot(t, z)
        axis[1].set_title("Z-position vs Time")
        axis[2].plot(t, theta)
        axis[2].set_title("Theta vs Time")
        plt.show()

if __name__ == "__main__":
    sim = TVCSimulation(mass=1, moment_of_inertia=0.1, center_of_mass_to_tvc=0.5, thrust=12,
                        time_step=0.1, start_time=0.0, stop_time=10.0, gravity=-9.81,
                        position_x=0, position_z=0, angular_position=0,
                        velocity_x=0, velocity_z=1, angular_velocity=0.5, 
                        acceleration_x=0, acceleration_z=0, angular_acceleration=0,
                        angle_limit=np.deg2rad(15), rate_limit=np.deg2rad(150),
                        K_p=0.07, K_i=0.01, K_d=0.01, pid_target=0, pid_min=-10, pid_max=10)
    sim.loop()