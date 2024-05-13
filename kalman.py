import numpy as np

class kalman:
    """
    Step 1a calculate prediction
    x^hat^-_t = A * x^hat^+_t-1 + B * u_t-1

    Step 1b update covariance of x error
    Sigma^-_x_t = A * Sigma^+_x_t-1 * A^T + Sigma_w

    Step 1c calculate output prediction
    y^hat_t = C * x^hat^-_t + D * u_t

    Step 2a calculate kalman gain
    L = Sigma_xy * Sigma_y^-1
    L = Sigma^-_x_t * C * [C * Sigma^-_x_t * C^T + Sigma_v]^-1

    Step 2b measurement update to state estimate
    x^hat^+_t = x^hat^-_t + L * (y_t - y^hat_t)

    Step 2c measurement update, impact on covariance of x error
    Sigma^+_x_t = Sigma^-_x_t - L * Sigma_y * L^T
    Sigma^+_x_t = Sigma^-_x_t - L * [C * Sigma^-_x_t * C^T + Sigma_v] * L^T
    """
    def __init__(self, A, B, C, D, Ts, sigma_w, sigma_v):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Ts = Ts
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v

        self.N_states = A.shape()[0]
        self.N_inputs = B.shape()[1]
        self.N_outputs = C.shape()[0]
        return

    def step1(self, x_plus_prev, u_prev, u_t, sigma_x_prev):
        # step 1a
        x_minus_t = self.A @ x_plus_prev + self.B @ u_prev

        # step 1b
        sigma_minus_x = self.A @ sigma_x_prev @ self.A.transpose() + self.sigma_w

        # step 1c
        y_hat_t = self.C @ x_minus_t + self.D @ u_t

        return x_minus_t, sigma_minus_x, y_hat_t

    def step2(self, x_minus_t, sigma_minus_x, y_hat_t, y_t):
        # step 2a calculate kalman gain
        sigma_y = (self.C @ sigma_minus_x @ self.C.transpose() + self.sigma_v)
        L = sigma_minus_x @ self.C.transpose() @ np.linalg.inv(sigma_y)

        # step 2b measurement update to state estimate
        x_plus_t = x_minus_t + L @ (y_t - y_hat_t)

        # step 2c measurement update to error covariance
        sigma_plus_x = sigma_minus_x - L @ sigma_y @ L.transpose()

        return x_plus_t, sigma_plus_x

    def simulate(self, x0, u, y):
        """
        x0 should be a column vector of initial state
        u should be a series of column vectors of inputs
        u.shape() should give (N_steps, N_inputs, 1)
        y should be a series of column vectors of measurements
        y.shape() should give (N_steps, N_outputs, 1)
        """
        N_steps = u.shape()[0] - 1
        x = np.zeros((N_steps+1, self.N_states, self.N_states))
        sigma_x = np.zeros((N_steps+1, self.N_states, self.N_states))
        x[0] = x0
        # assume that the initial state is known exactly:
        sigma_x[0] = np.zeros((self.N_states, self.N_states))

        for t in range(1, N_steps+1):
            x_minus_t, sigma_minus_x, y_hat_t = self.step1(x[t-1], u[t-1], u[t], sigma_x[t-1])
            x_plus_t, sigma_plus_x = self.step2(x_minus_t, sigma_minus_x, y_hat_t, y[t])
            x[t] = x_plus_t
            sigma_x[t] = sigma_plus_x

        return x, sigma_x

