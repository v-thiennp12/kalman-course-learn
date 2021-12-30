import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE']         = [12,12]

options['DRIVE_IN_CIRCLE']  = True
    # If False, measurements will be x,y.
    # If True, measurements will be x,y, and current angle of the car.
    # Required if you want to pass the driving in circle.
options['MEASURE_ANGLE']    = True
options['RECIEVE_INPUTS']   = True

class KalmanFilter:
    def __init__(self):
        # Initial State
        self.x = np.matrix([[0.],
                            [0.],
                            [0.],
                            [0.],
                            [0.]]) # [x y v theta theta_dot]

        # external force
        self.u = np.matrix([[0.],
                            [0.],
                            [0.],
                            [0.],
                            [0.]])

        # Uncertainity Matrix
        self.P = np.matrix([[1000., 0., 0., 0., 0.],
                            [0., 1000., 0., 0., 0.],
                            [0., 0., 1000., 0., 0.],
                            [0., 0., 0., 1000., 0.],
                            [0., 0., 0., 0., 1000.]])

        # Next State Function
        self.F = np.matrix([[1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0.]])

        # Measurement Function
        self.H = np.matrix([[1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 1., 0.]])

        # Measurement Uncertainty
        self.R = np.matrix([[5.,    0.,     0.],
                            [0.,    5.,     0.],
                            [0.,    0.,     5.]])

        # Identity Matrix
        self.I = np.matrix([[1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 1.]])

    def predict(self, dt):
        #add noise to uncertainty to do not overfit Kalman filter confidence
        self.P[0, 0] += 0.1
        self.P[1, 1] += 0.1
        self.P[2, 2] += 0.1
        self.P[3, 3] += 0.1
        self.P[4, 4] += 0.1

        # #add dt to self.F transition matrix        
        # self.F.item(0, 2)  = dt # x = x_prev + x_dot*dt
        # self.F.item(1, 3)  = dt # y = y_prev + y_dot*dt
        # x_dot = v*cos(theta)

        self.F[0, 2] = dt*np.cos(self.x[3,0])
        self.F[1, 2] = dt*np.sin(self.x[3,0])
        self.F[3, 4] = dt

        # state = F*prev_state
        self.x      = self.F*self.x + self.u
        #uncertainty matrix # P = F*P*F^T
        self.P      = self.F*self.P*np.transpose(self.F)
        
        return #

    def measure_and_update(self,measurements, dt):
        #measurement update
        Z = np.matrix(measurements)
        y = np.transpose(Z) - self.H*self.x                 #error {measure | estimated_state}        
        S = self.H*self.P*np.transpose(self.H) + self.R     #measurement uncertainty accumulation       
        K = self.P*np.transpose(self.H)*np.linalg.inv(S)    #disturbance estimation

        #update state
        self.x = self.x + K*y                               #update state
        self.P = (self.I - K*self.H)*self.P                 #update uncertainty
        
        return [self.x[0], self.x[1]]

    def recieve_inputs(self, u_steer, u_pedal):

        # self.u[2, 0] = (-self.x[2] + 1.0*u_pedal)/2.0*dt
        # self.u[3, 0] = u_steer*dt

        self.x[2, 0] = u_pedal
        self.x[4, 0] = u_steer

        return #[self.x[0], self.x[1]]

sim_run(options,KalmanFilter)