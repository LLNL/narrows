"""
Class for solving the 1D neutron transport equation in a slab geometry.
Author: Kyle Bilton
Date: 11/2018
"""
import numpy as np
import torch

from .writer import write


class ANNSlabSolver(object):
    """
    Solver for the slab geometry neutron transport equation using neural
    networks. The ANN uses a single hidden layer.
    """

    def __init__(self, N, n_nodes, edges, sigma_t, sigma_s0, sigma_s1, source,
                 gamma_l=50, gamma_r=50, learning_rate=1e-3, eps=1e-8,
                 use_weights=False, tensorboard=False, interval=500,
                 gpu=False, ahistory=False, hinterval=1):
        """
        Parameters
        ==========
        N : int
            Order of the Legendre-Gauss quadratures.
        n_nodes : int
            Number of nodes in the hidden layer of the neural net.
        edges : numpy array
            The edges of the spatial discretization.
        sigma_t : numpy array
            Total macroscopic cross section.
        sigma_s0 : numpy array
            First term in the scattering cross section expansion.
        sigma_s1 : numpy array
            Second term in the scattering cross section expansion.
        source : numpy array
            Magnitude of the external source.
        gamma_l : int
            Left boundary regularizer coefficient.
        gamma_r : int
            Right boundary regularizer coefficient.
        learning_rate : float
            Learning rate of the Adam optimizer.
        eps : float
            Convergence criterion comparison quantity.
        use_weights : bool
            Use updated residual weights in minimizing the loss.
        tensorboard : bool
            Use tensorboard.
        interval : int
            Interval at which loss is printed.
        gpu : bool
            Run on gpu
        ahistory : bool
            Record loss and flux arrays every hinterval iterations
        hinterval : int
            The number of interations between recordings
        """

        self.device = torch.device('cpu')
        self.gpu = False
        if gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
                self.gpu = True
            else:
                write('terse', 'Skipping gpu')
        ########################################
        # Angular Meshing
        ########################################

        self.N = N

        # Get the Legendre-Gauss quadratures
        mu, w = np.polynomial.legendre.leggauss(N)
        self.mu = mu
        self.w = w
        # Put the quadratures into tensors
        mu_t = torch.tensor(mu.astype(np.float32), device=self.device)
        w_t = torch.tensor(w.astype(np.float32), device=self.device)
        self.mu_t = mu_t
        self.w_t = w_t

        # Turn the edges into a PyTorch tensor, stacked as a column
        # (hence the [:,None])
        self.z = torch.from_numpy(edges[:, None]).to(self.device)
        # In this case, we want to track the gradients with respect to z,
        # so specify that here
        z_t = torch.autograd.Variable(self.z, requires_grad=True)
        self.z_t = z_t

        ########################################
        # Material properties
        ########################################
        self.sigma_t = torch.from_numpy(sigma_t).unsqueeze(1).repeat(1, N)
        self.sigma_t = self.sigma_t.to(self.device)

        self.sigma_s0 = torch.from_numpy(sigma_s0).unsqueeze(1).repeat(1, N)
        self.sigma_s0 = self.sigma_s0.to(self.device)

        self.sigma_s1 = torch.from_numpy(sigma_s1).unsqueeze(1).repeat(1, N)
        self.sigma_s1 = self.sigma_s1.to(self.device)

        # Set data on the external source
        self.Q_t = torch.from_numpy(source).unsqueeze(1).repeat(1, N)
        self.Q_t = self.Q_t.to(self.device)

        ########################################
        # Neural Network Parameters
        ########################################

        self.n_nodes = n_nodes

        summary_writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                summary_writer = SummaryWriter()
            except Exception as e:
                write('terse', 'Skipping tensorboard')
                write('moderate', e, error=True)

        self._build_model(summary_writer)

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)

        # Set regularization coefficients
        self.gamma_l = gamma_l
        self.gamma_r = gamma_r

        # Set the convergence criterion
        self.eps = eps

        self.use_weights = use_weights

        num_points_z = len(self.z)
        self.gamma = \
            torch.ones(num_points_z, device=self.device).reshape(-1, 1)

        ones = torch.ones(num_points_z, device=self.device).reshape(-1, 1)
        self.r_squared_opt = (eps / num_points_z * ones)

        self.interval = interval

        self.ahistory = ahistory

        self.hinterval = hinterval

    def _build_model(self, summary_writer=None):
        """
        Build neural network model.
        """
        model = torch.nn.Sequential(torch.nn.Linear(1, self.n_nodes),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(self.n_nodes, self.N),)
        if summary_writer:
            summary_writer.add_graph(model, self.z_t)

        if self.gpu:
            self.model = model.cuda()
        else:
            self.model = model

    def _loss(self, y_pred, z):
        """
        Loss function for the network
        """

        # Calculate the isotropic flux
        phi_0 = self._compute_scalar_flux(psi=y_pred).reshape(-1, 1)
        phi_1 = torch.matmul(y_pred, self.mu_t * self.w_t).reshape(-1, 1)

        # Create a placeholder for the gradient
        grad = torch.empty_like(y_pred, device=self.device)

        # Compute the gradient of each output with respect to the input
        for idx in range(y_pred.shape[1]):
            grad_outputs = y_pred.data.new(y_pred.shape[0]).fill_(1)
            g, = torch.autograd.grad(y_pred[:, idx], z,
                                     grad_outputs=grad_outputs,
                                     create_graph=True)
            grad[:, idx] = g.flatten()

        # Compute the loss
        l1 = (self.mu_t * grad + self.sigma_t * y_pred
              - 0.5 * (self.sigma_s0 * phi_0
                       + 3 * self.mu_t * self.sigma_s1 * phi_1)
              - 0.5 * self.Q_t)**2

        spatial_loss = l1.sum(1)
        self.r_squared = spatial_loss.reshape(-1, 1)
        loss = 0.5 * torch.dot(self.gamma.flatten(),
                               self.r_squared.flatten()).reshape(1)

        # Use the previous squared error as the weights
        if self.use_weights:
            ones = torch.ones_like(self.r_squared, device=self.device)
            log1 = torch.log(self.r_squared_opt, device=self.device)
            log2 = torch.log(self.r_squared, device=self.device)
            rho = torch.max(ones, torch.abs(log1 / log2))

            ones = torch.ones_like(rho, device=self.device)
            omega = phi_0 / phi_0.max()
            self.gamma = torch.max(ones, rho / omega, device=self.device)

        # Add a penalty relating to the boundary conditions
        loss += (0.5 * self.gamma_l *
                 ((y_pred[0, self.N//2:])**2).sum().reshape(1))
        loss += (0.5 * self.gamma_r *
                 ((y_pred[-1, :self.N//2])**2).sum().reshape(1))

        return torch.sum(loss), spatial_loss.detach().numpy()

    def train(self, num_iterations_estimate=2**20):
        """
        Train the neural network.
        """

        loss_history = np.zeros(num_iterations_estimate)

        if self.ahistory:
            shape = (num_iterations_estimate // self.hinterval, len(self.z))
            spatial_loss_history = np.zeros(shape)
            flux_history = np.zeros(shape)

        prev_loss = 1e6
        it = 0
        while True:
            # First, compute the estimate, which is known as the forward pass
            y_pred = self.model(self.z_t)

            # Compute the loss between the prediction and true value
            loss, spatial_loss = self._loss(y_pred, self.z_t)
            loss_history[it] = loss

            # Inspect the value of the loss
            if it % self.interval == 0:
                write('moderate', f'Iter {it}: {loss.item()}')

            if self.ahistory and (it % self.hinterval == 0):
                recording_num = it // self.hinterval
                spatial_loss_history[recording_num] = spatial_loss
                flux_history[recording_num] = self._compute_scalar_flux(
                        psi=y_pred, numpy=True)

            self.optimizer.zero_grad()

            # Peform the backwards propagation of gradients
            loss.backward(retain_graph=True)

            # Now update the parameters using the newly-calculated gradients
            self.optimizer.step()

            loss = loss.item()

            err = np.abs(loss - prev_loss)
            prev_loss = loss

            if err < self.eps:
                write('moderate', f'Iter {it}: {loss}')
                break
            it += 1

        result = {'loss_history': np.trim_zeros(loss_history),
                  'spatial_loss': spatial_loss}

        if self.ahistory:
            size = it // self.hinterval
            result['spatial_loss_history'] = spatial_loss_history[:size]
            result['flux_history'] = flux_history[:size]

        return result

    def _compute_scalar_flux(self, z=None, psi=None, numpy=False):
        """
        Compute the scalar flux at points z.

        Parameters
        ==========
        z : torch tensor, shape (1, n_points_z)
            Spatial variable to compute flux at.
        numpy : bool, default=False
            If True, return a numpy array. Otherwise return a torch tensor.

        Returns
        =======
        phi_0 : array-like, shape (n_points_z)
            The predicted scalar flux from the neural network at each z-point.
        """

        if psi is None:
            psi = self.model(z)

        phi_0 = torch.matmul(psi, self.w_t)

        if numpy:
            if self.gpu:
                return phi_0.detach().to('cpu').numpy()
            else:
                return phi_0.detach().numpy()
        else:
            return phi_0

    def predict(self, z=None):
        """
        Predict the flux at spatial positions z. If z is None, use the
        same spatial variables that were used in training the network.

        Parameters
        ==========
        z : array-like, shape (n_points_z)
            Spatial variable to compute flux at.

        Returns
        =======
        phi : array-like, shape (n_points_z)
            The predicted flux from the neural network at each z-point.
        """

        # Use the existing spatial values if none are provided
        if z is None:
            return self._compute_scalar_flux(z=self.z_t, numpy=True)

        # Otherwise, compute flux on the new spatial values
        arg = torch.tensor(z[:, None], device=self.device)
        return self._compute_scalar_flux(z=arg, numpy=True)
