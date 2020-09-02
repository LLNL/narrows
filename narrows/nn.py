import numpy as np
import torch

from .writer import write

LARGE_DOUBLE = 1e50


class ANNSlabSolver(object):
    '''
    Solver for the slab geometry neutron transport equation using neural
    networks. The ANN uses a single hidden layer.
    '''

    def __init__(self, N, n_nodes, edges, sigma_t, sigma_s0, sigma_s1, source,
                 gamma_l=50, gamma_r=50, learning_rate=1e-3, eps=1e-8,
                 tensorboard=False, interval=500, gpu=False, ahistory=False,
                 hinterval=1, max_num_iter=100000):
        '''
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
        max_num_iter : int
            The maximum number of iterations before we quit trying to converge
        '''

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

        # Put the quadratures into tensors
        self.mu = torch.tensor(mu.astype(np.float32), device=self.device)
        self.w = torch.tensor(w.astype(np.float32), device=self.device)

        # Turn the edges into a PyTorch tensor, stacked as a column
        # (hence the [:, None]), and track the gradients with respect to z
        self.z = torch.tensor(edges[:, None],
                              requires_grad=True).to(self.device)

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

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)

        self.gamma_l = gamma_l
        self.gamma_r = gamma_r

        self.eps = eps

        self.interval = interval

        self.ahistory = ahistory

        self.hinterval = hinterval

        self.max_num_iter = max_num_iter

    def _build_model(self, summary_writer=None):
        '''
        Build neural network model.
        '''
        model = torch.nn.Sequential(torch.nn.Linear(1, self.n_nodes),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(self.n_nodes, self.N),)
        if summary_writer:
            summary_writer.add_graph(model, self.z)

        if self.gpu:
            self.model = model.cuda()
        else:
            self.model = model

    def _loss(self, y_pred, z):
        '''
        Loss function for the network
        '''

        # Calculate the isotropic flux
        phi_0 = self._compute_scalar_flux(psi=y_pred).reshape(-1, 1)
        phi_1 = torch.matmul(y_pred, self.mu * self.w).reshape(-1, 1)

        # Create a placeholder for the gradient
        grad = torch.empty_like(y_pred, device=self.device)

        # Compute the gradient of each output with respect to the input
        for angle_index in range(y_pred.shape[1]):
            grad_outputs = y_pred.data.new(y_pred.shape[0]).fill_(1)
            g, = torch.autograd.grad(y_pred[:, angle_index], z,
                                     grad_outputs=grad_outputs,
                                     create_graph=True)
            grad[:, angle_index] = g.flatten()

        # Compute the loss
        angular_loss = (self.mu * grad + self.sigma_t * y_pred
                        - 0.5 * (self.sigma_s0 * phi_0
                                 + 3 * self.mu * self.sigma_s1 * phi_1)
                        - 0.5 * self.Q_t)**2

        spatial_loss = angular_loss.sum(1)
        loss = 0.5 * spatial_loss.sum(0).reshape(1)

        # Add a penalty relating to the boundary conditions
        loss += (0.5 * self.gamma_l *
                 ((y_pred[0, self.N//2:])**2).sum().reshape(1))
        loss += (0.5 * self.gamma_r *
                 ((y_pred[-1, :self.N//2])**2).sum().reshape(1))

        if self.gpu:
            sloss = spatial_loss.detach().to('cpu').numpy()
        else:
            sloss = spatial_loss.detach().numpy()

        return torch.sum(loss), sloss

    def _record_history(self, recording_index, spatial_loss, y_pred,
                        spatial_loss_history, flux_history):
        '''
        Write history to arrays

        Parameters
        ==========
        recording_index : int
            The index of this recording.
        spatial_loss : array-like, shape (n_points_z)
            The spatial distribution of the loss.
        y_pred : torch tensor, shape (n_points_z, num_ordinates)
            Angular flux estimate
        spatial_loss_history : array-like, shape (E, n_points_z) where E is an
                               estimate of the total number of recordings
            The spatial loss at each recorded iteration.
        flux_history : array-like, shape (E, n_points_z) where E is an
                       estimate of the total number of recordings
            The flux at each recorded iteration.
        '''
        spatial_loss_history[recording_index] = spatial_loss
        flux_history[recording_index] = self._compute_scalar_flux(
                psi=y_pred, numpy=True)

    def train(self, num_iterations_estimate=2**20):
        '''
        Train the neural network.
        '''

        loss_history = np.zeros(num_iterations_estimate)

        if self.ahistory:
            num_recordings_estimate = (num_iterations_estimate //
                                       self.hinterval)
            num_recordings_estimate += 2  # First and last iteration
            shape = (num_recordings_estimate, len(self.z))
            spatial_loss_history = np.zeros(shape)
            flux_history = np.zeros(shape)

        prev_loss = LARGE_DOUBLE
        iteration_index = 0
        while True:
            # First, compute the estimate, which is known as the forward pass
            y_pred = self.model(self.z)

            # Compute the loss between the prediction and true value
            loss, spatial_loss = self._loss(y_pred, self.z)
            loss_history[iteration_index] = loss

            if iteration_index % self.interval == 0:
                write('moderate', f'Iter {iteration_index}: {loss.item()}')
                loss_was_printed = True
            else:
                loss_was_printed = False

            if self.ahistory and (iteration_index % self.hinterval == 0):
                recording_index = iteration_index // self.hinterval
                self._record_history(recording_index, spatial_loss, y_pred,
                                     spatial_loss_history, flux_history)
                history_was_recorded = True
            else:
                history_was_recorded = False

            self.optimizer.zero_grad()

            # Peform the backwards propagation of gradients
            loss.backward(retain_graph=True)

            # Now update the parameters using the newly-calculated gradients
            self.optimizer.step()

            loss = loss.item()

            err = np.abs(loss - prev_loss)
            prev_loss = loss

            if err < self.eps or iteration_index == self.max_num_iter:
                if not loss_was_printed:
                    write('moderate', f'Iter {iteration_index}: {loss}')
                if self.ahistory and not history_was_recorded:
                    recording_index = (iteration_index // self.hinterval) + 1
                    self._record_history(recording_index, spatial_loss, y_pred,
                                         spatial_loss_history, flux_history)
                if iteration_index == self.max_num_iter:
                    write('terse', 'Maximum number of iterations achieved')
                break
            iteration_index += 1

        result = {'loss_history': np.trim_zeros(loss_history),
                  'spatial_loss': spatial_loss}

        if self.ahistory:
            size = (iteration_index // self.hinterval) + 2
            result['spatial_loss_history'] = spatial_loss_history[:size]
            result['flux_history'] = flux_history[:size]

        return result

    def _compute_scalar_flux(self, z=None, psi=None, numpy=False):
        '''
        Compute the scalar flux at points z.

        Parameters
        ==========
        z : torch tensor, shape (1, n_points_z)
            Spatial variable to compute flux at.
        psi : torch tensor, shape (n_points_z, num_ordinates)
            Angular flux
        numpy : bool
            If True, return a numpy array. Otherwise return a torch tensor.

        Returns
        =======
        phi_0 : array-like, shape (n_points_z)
            The predicted scalar flux from the neural network at each z-point.
        '''

        if psi is None:
            psi = self.model(z)

        phi_0 = torch.matmul(psi, self.w)

        if numpy:
            if self.gpu:
                return phi_0.detach().to('cpu').numpy()
            else:
                return phi_0.detach().numpy()
        else:
            return phi_0

    def predict(self, z=None):
        '''
        Predict the flux at spatial positions z. If z is None, use the
        same spatial variables that were used in training the network.

        Parameters
        ==========
        z : array-like, shape (n_points_z)
            Spatial variable to compute flux at.

        Returns
        =======
        phi : array-like, shape (n_points_z)
            The predicted scalar flux from the neural network at each z-point.
        '''

        # Use the existing spatial values if none are provided
        if z is None:
            return self._compute_scalar_flux(z=self.z, numpy=True)

        # Otherwise, compute flux on the new spatial values
        arg = torch.tensor(z[:, None], device=self.device)
        return self._compute_scalar_flux(z=arg, numpy=True)
