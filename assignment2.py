import numpy as np
from sim.sim1d import sim_run

# Simulator options.
options                     = {}
options['FIG_SIZE']         = [8,8]
options['CONSTANT_SPEED']   = True

class KalmanFilter:
    def __init__(self):
        self.v = 0
        self.prev_time = 0
        # Initial State # state = [x x_dot]
        self.x = np.matrix([[0.],
                            [0.]])

        # Uncertainity Matrix
        self.P = np.matrix([[1000, 0],
                            [0.,  1000]])

        # Next State Function
        self.F = np.matrix([[1., 1000.],
                            [0., 1.]])

        # Measurement Function
        self.H = np.matrix([[1., 0.]])

        # Measurement Uncertainty
        self.R = np.matrix([[0.01]])

        # Identity Matrix
        self.I = np.matrix([[1., 0.],
                            [0., 1.]])

    def predict(self, t):
        # Calculate dt.
        dt          = t - self.prev_time        
        # Put dt into the state transition matrix
        self.F[0,1] = dt # x = x_prev + x_dot*dt
        # state = F*prev_state
        self.x      = self.F*self.x
        #uncertainty matrix # P = F*P*F^T
        self.P      = self.F*self.P*np.transpose(self.F)
        print(self.P, '\n')
        return self.x[0,0]

    def measure_and_update(self,measurements,t):        
        dt          = t - self.prev_time                    #calculate dt
        #transition update
        self.F[0,1] = dt

        #measurement update
        Z = np.matrix(measurements)
        y = np.transpose(Z) - self.H*self.x                 #error {measure | estimated_state}        
        S = self.H*self.P*np.transpose(self.H) + self.R     #measurement uncertainty accumulation       
        K = self.P*np.transpose(self.H)*np.linalg.inv(S)    #disturbance estimation

        #update state
        self.x = self.x + K*y                               #update state
        self.P = (self.I - K*self.H)*self.P                 #update uncertainty
        #add noise to uncertainty to do not overfit Kalman filter confidence
        self.P[0, 0] += 0.1
        self.P[1, 1] += 0.1

        self.v = self.x[1,0]                                #get x_dot from state matrix
        self.prev_time = t                                  #important, update previous time
        return

sim_run(options,KalmanFilter)