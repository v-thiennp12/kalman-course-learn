import numpy as np
from sim.sim2d_prediction import sim_run

# Simulator options.
options                     = {}
options['FIG_SIZE']         = [8,8]
options['ALLOW_SPEEDING']   = True

class KalmanFilter:
    def __init__(self):
        # Initial State
        self.x = np.matrix([[55.],
                            [3.],
                            [5.],
                            [0.]])

        # external force
        self.u = np.matrix([[0.],
                            [0.],
                            [0.],
                            [0.]])

        # Uncertainity Matrix
        self.P = np.matrix([[0., 0., 0., 0.],
                            [0., 0., 0., 0.],
                            [0., 0., 0., 0.],
                            [0., 0., 0., 0.]])

        # Next State Function
        self.F = np.matrix([[1., 0., 0.1, 0.],
                            [0., 1., 0., 0.1],
                            [0., 0., 1., 0],
                            [0., 0., 0., 1.]])

        # Measurement Function
        self.H = np.matrix([[1., 0., 0., 0.],
                            [0., 1., 0., 0.]])

        # Measurement Uncertainty
        self.R = np.matrix([[5., 0.0],
                            [0.,  5.]])

        # Identity Matrix
        self.I = np.matrix([[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]])

    def predict(self, dt):
        #add noise to uncertainty to do not overfit Kalman filter confidence
        self.P[0, 0] += 0.1
        self.P[1, 1] += 0.1
        self.P[2, 2] += 0.1
        self.P[3, 3] += 0.1

        # state = F*prev_state
        self.x      = self.F*self.x #+ self.u
        #uncertainty matrix # P = F*P*F^T
        self.P      = self.F*self.P*np.transpose(self.F)

        return
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

    def predict_red_light(self,light_location):
        light_duration  = 3
        F_new           = np.copy(self.F)

        F_new[0,2]      = light_duration
        F_new[1,3]      = light_duration

        x_new           = F_new*self.x #+ self.u

        if x_new[0] < light_location:
            return [False, x_new[0]]
        else:
            return [True, x_new[0]]

    def predict_red_light_speed(self, light_location):
        light_duration  = 3
        F_new           = np.copy(self.F)
        
        #accel time from current speed v m/s to v+1.5 m/s
        accel_duration  = +1 #s
        #prediction at +1s
        F_new[0,2]      = accel_duration
        F_new[1,3]      = accel_duration
        self.u[2]       = 1.5 
        x_new           = F_new*self.x + self.u

        #prediction at +3s from +1s previously predicted        
        F_new[0,2]      = light_duration - accel_duration
        F_new[1,3]      = light_duration - accel_duration
        self.u[2]       = 0
        x_new           = F_new*x_new + self.u # predict from previous state, so x_new not self.x)

        if x_new[0] < light_location:
            return [False, x_new[0]]
        else:
            return [True, x_new[0]]

# sim_run(options,KalmanFilter,0)

for i in range(0,5):
    sim_run(options,KalmanFilter,i)
