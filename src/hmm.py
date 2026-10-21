"""
hmm.py
------
Hidden Markov Model — implemented from scratch using NumPy.

Hidden States:
    0 = FRESH
    intermediate states = DEGRADING
    final state = CRITICAL

Observations: continuous (Gaussian emissions per state)

Math reference:
    Rabiner (1989) "A Tutorial on Hidden Markov Models"
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


STATE_NAMES = {0: 'FRESH', 1: 'DEGRADING', 2: 'CRITICAL'}
N_STATES = 3
MIN_COVARIANCE_EIGENVALUE = 1e-3


class GaussianHMM:
    """
    HMM with Gaussian (continuous) emissions.

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 3)
    n_iter : int
        Number of Baum-Welch EM iterations
    tol : float
        Convergence threshold on log-likelihood
    """

    def __init__(self, n_states: int = 3, n_iter: int = 50, tol: float = 1e-4):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol

        # Model parameters (learned during training)
        self.pi = None       # Initial state distribution  shape: (n_states,)
        self.A = None        # Transition matrix           shape: (n_states, n_states)
        self.mu = None       # Emission means              shape: (n_states, n_features)
        self.sigma = None    # Emission covariances        shape: (n_states, n_features, n_features)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_params(self, X: np.ndarray):
        """
        Initialize states in the order of the final observation feature.
        """
        n_features = X.shape[1]

        self.pi = np.zeros(self.n_states)
        self.pi[0] = 1.0

        self.A = np.zeros((self.n_states, self.n_states))
        for state in range(self.n_states - 1):
            self.A[state, state] = 0.8
            self.A[state, state + 1] = 0.2
        self.A[-1, -1] = 1.0

        ordered = X[np.argsort(X[:, -1])]
        chunks = np.array_split(ordered, self.n_states)
        self.mu = np.array([chunk.mean(axis=0) for chunk in chunks])
        self.sigma = np.array([
            self._regularize_covariance(
                np.atleast_2d(np.cov(chunk.T)) if len(chunk) > 1 else np.eye(n_features)
            )
            for chunk in chunks
        ])

    @staticmethod
    def _regularize_covariance(covariance: np.ndarray) -> np.ndarray:
        """Symmetrize a covariance matrix and bound its eigenvalues away from zero."""
        covariance = (covariance + covariance.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, MIN_COVARIANCE_EIGENVALUE)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    # ------------------------------------------------------------------
    # Emission probability
    # ------------------------------------------------------------------

    def _log_emission_prob(self, obs: np.ndarray) -> np.ndarray:
        """Return Gaussian log-emission probabilities with shape (T, n_states)."""
        log_B = np.zeros((len(obs), self.n_states))
        for j in range(self.n_states):
            log_B[:, j] = multivariate_normal.logpdf(
                obs, mean=self.mu[j], cov=self.sigma[j]
            )
        if not np.isfinite(log_B).all():
            raise ValueError('Gaussian emissions produced non-finite log probabilities.')
        return log_B

    # ------------------------------------------------------------------
    # Forward Algorithm  O(T * N^2)
    # ------------------------------------------------------------------

    def _forward_log(self, log_B: np.ndarray) -> tuple[np.ndarray, float]:
        """Run the forward algorithm in log space and return log-alpha values."""
        log_A = np.full_like(self.A, -np.inf)
        positive_transitions = self.A > 0
        log_A[positive_transitions] = np.log(self.A[positive_transitions])
        log_pi = np.full_like(self.pi, -np.inf)
        positive_initial = self.pi > 0
        log_pi[positive_initial] = np.log(self.pi[positive_initial])

        log_alpha = np.empty((len(log_B), self.n_states))
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, len(log_B)):
            log_alpha[t] = log_B[t] + logsumexp(
                log_alpha[t - 1][:, None] + log_A, axis=0
            )

        return log_alpha, float(logsumexp(log_alpha[-1]))

    def filter(self, obs: np.ndarray) -> np.ndarray:
        """Return P(state_t | observations through t), without future observations."""
        if obs.ndim != 2 or len(obs) == 0:
            raise ValueError('Filtering requires a non-empty 2D observation sequence.')
        log_alpha, _ = self._forward_log(self._log_emission_prob(obs))
        return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))

    # ------------------------------------------------------------------
    # Viterbi Algorithm  O(T * N^2)
    # ------------------------------------------------------------------

    def viterbi(self, obs: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding: find the most likely hidden state sequence.

            δ_t(j) = max_{q_1,...,q_{t-1}} P(q_1,...,q_{t-1}, q_t=j, o_1,...,o_t | λ)

        Recursion:
            δ_1(j) = π_j * B_j(o_1)
            δ_t(j) = max_i [δ_{t-1}(i) * A_{ij}] * B_j(o_t)
            ψ_t(j) = argmax_i [δ_{t-1}(i) * A_{ij}]   ← backpointer

        Backtrack through ψ to recover the best path.

        Args:
            obs: shape (T, n_features)
        Returns:
            states: shape (T,) — most likely state at each timestep
        """
        T = obs.shape[0]
        N = self.n_states
        # Use log probabilities to avoid underflow
        log_A = np.where(self.A > 0, np.log(np.maximum(self.A, 1e-300)), -np.inf)
        log_pi = np.where(self.pi > 0, np.log(np.maximum(self.pi, 1e-300)), -np.inf)
        log_B = self._log_emission_prob(obs)

        delta = np.zeros((T, N))   # max log-prob up to t ending in state j
        psi = np.zeros((T, N), dtype=int)  # backpointers

        # Initialization
        delta[0] = log_pi + log_B[0]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                candidates = delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + log_B[t, j]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states

    # ------------------------------------------------------------------
    # Baum-Welch (EM Algorithm)
    # ------------------------------------------------------------------

    def fit(self, sequences: list):
        """
        Baum-Welch algorithm: learn A, B (mu, sigma), pi from observed sequences.

        E-step: compute γ_t(i) and ξ_t(i,j) using forward-backward.
        M-step: update A, mu, and sigma while keeping pi fixed at [1, 0, ...].

        γ_t(i) = P(q_t=i | O, λ)  ← prob of being in state i at time t
        ξ_t(i,j) = P(q_t=i, q_{t+1}=j | O, λ)  ← prob of transition i→j at t

        Args:
            sequences: list of np.arrays, each shape (T_k, n_features)
        """
        if not sequences:
            raise ValueError('Cannot fit an HMM without training sequences.')

        sequences = [np.asarray(sequence, dtype=float) for sequence in sequences]
        if any(sequence.ndim != 2 or len(sequence) == 0 for sequence in sequences):
            raise ValueError('Training sequences must be non-empty 2D arrays.')
        n_features = sequences[0].shape[1]
        if any(sequence.shape[1] != n_features for sequence in sequences):
            raise ValueError('All training sequences must have the same feature count.')
        if any(not np.isfinite(sequence).all() for sequence in sequences):
            raise ValueError('Training sequences must contain only finite observations.')

        all_obs = np.vstack(sequences)
        self._initialize_params(all_obs)

        prev_ll = -np.inf

        print(f"Training HMM ({self.n_states} states, {n_features} features)...")

        for iteration in range(self.n_iter):
            # Accumulators for M-step (summed across all sequences)
            sum_xi = np.zeros((self.n_states, self.n_states))
            sum_gamma = np.zeros(self.n_states)
            sum_gamma_obs = np.zeros((self.n_states, n_features))
            sum_gamma_obs_outer = np.zeros((self.n_states, n_features, n_features))
            total_ll = 0.0
            log_A = np.full_like(self.A, -np.inf)
            positive_transitions = self.A > 0
            log_A[positive_transitions] = np.log(self.A[positive_transitions])

            # ---- E-step ----
            for obs in sequences:
                log_B = self._log_emission_prob(obs)
                log_alpha, ll = self._forward_log(log_B)
                total_ll += ll

                log_beta = np.zeros_like(log_alpha)
                for t in range(len(obs) - 2, -1, -1):
                    log_beta[t] = logsumexp(
                        log_A
                        + log_B[t + 1][None, :]
                        + log_beta[t + 1][None, :],
                        axis=1,
                    )

                log_gamma = log_alpha + log_beta
                log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
                gamma = np.exp(log_gamma)

                if len(obs) > 1:
                    log_xi = (
                        log_alpha[:-1, :, None]
                        + log_A[None, :, :]
                        + log_B[1:, None, :]
                        + log_beta[1:, None, :]
                    )
                    log_xi -= logsumexp(log_xi, axis=(1, 2), keepdims=True)
                    sum_xi += np.exp(log_xi).sum(axis=0)

                # Accumulate
                sum_gamma += gamma.sum(axis=0)

                for j in range(self.n_states):
                    w = gamma[:, j]  # shape (T,)
                    sum_gamma_obs[j] += (w[:, None] * obs).sum(axis=0)
                    sum_gamma_obs_outer[j] += obs.T @ (w[:, None] * obs)

            # ---- M-step ----
            updated_A = np.zeros_like(self.A)
            for state in range(self.n_states - 1):
                allowed = [state, state + 1]
                transition_counts = sum_xi[state, allowed]
                count_sum = transition_counts.sum()
                updated_A[state, allowed] = (
                    transition_counts / count_sum if count_sum > 0 else self.A[state, allowed]
                )
            updated_A[-1, -1] = 1.0
            self.A = updated_A

            for j in range(self.n_states):
                if sum_gamma[j] <= 1e-12:
                    continue
                self.mu[j] = sum_gamma_obs[j] / sum_gamma[j]
                second_moment = sum_gamma_obs_outer[j] / sum_gamma[j]
                covariance = second_moment - np.outer(self.mu[j], self.mu[j])
                self.sigma[j] = self._regularize_covariance(covariance)

            if not np.isfinite(total_ll):
                raise ValueError('HMM training produced a non-finite log-likelihood.')

            # Convergence check
            delta_ll = total_ll - prev_ll
            print(f"  Iter {iteration+1:3d} | log-likelihood: {total_ll:.4f} | delta: {delta_ll:.6f}")

            if abs(delta_ll) < self.tol and iteration > 5:
                print(f"  Converged at iteration {iteration+1}.")
                break

            prev_ll = total_ll

        print("Training complete.")
        print(f"\nLearned Transition Matrix A:\n{np.round(self.A, 3)}")
        print(f"\nLearned Initial Distribution pi: {np.round(self.pi, 3)}")

    def decode(self, obs: np.ndarray) -> list:
        """
        Return named causal, forward-filtered state predictions.

        Returns list of state names e.g. ['FRESH', 'FRESH', 'DEGRADING', 'CRITICAL']
        """
        state_ids = self.filter(obs).argmax(axis=1)
        if self.n_states == 3:
            names = STATE_NAMES
        else:
            names = {
                state: (
                    'FRESH' if state == 0
                    else 'CRITICAL' if state == self.n_states - 1
                    else f'STATE_{state}'
                )
                for state in range(self.n_states)
            }
        return [names[int(state)] for state in state_ids]

    def predict_pit_laps(self, obs: np.ndarray, threshold_state: int | None = None) -> list:
        """
        Return indices where the filtered state first enters the selected state.

        Args:
            obs             : observation sequence shape (T, n_features)
            threshold_state : state considered critical (defaults to the final state)
        Returns:
            list of lap indices where a pit stop is predicted
        """
        threshold_state = self.n_states - 1 if threshold_state is None else threshold_state
        if threshold_state < 0 or threshold_state >= self.n_states:
            raise ValueError(f'State index {threshold_state} is outside this model.')
        states = self.filter(obs).argmax(axis=1)
        pit_laps = []
        for t in range(len(states)):
            if states[t] == threshold_state and (t == 0 or states[t-1] != threshold_state):
                pit_laps.append(t)
        return pit_laps


if __name__ == '__main__':
    # Sanity check with synthetic data
    np.random.seed(42)

    def make_stint(n_laps=20, n_features=4):
        """Simulate a stint: features drift as tyre degrades."""
        obs = []
        for i in range(n_laps):
            deg = i / n_laps
            lap = [
                deg * 3 + np.random.randn() * 0.3,   # lap_time_delta grows
                deg + np.random.randn() * 0.1,         # tyre_age_norm
                deg * 1.5 + np.random.randn() * 0.2,  # sector3_delta
                deg * 2 + np.random.randn() * 0.2,    # rolling_degradation
            ]
            obs.append(lap)
        return np.array(obs)

    sequences = [make_stint(20) for _ in range(10)]

    model = GaussianHMM(n_states=3, n_iter=20)
    model.fit(sequences)

    test_seq = make_stint(20)
    states = model.decode(test_seq)
    print(f"\nDecoded states for test stint:\n{states}")

    pit_laps = model.predict_pit_laps(test_seq)
    print(f"Predicted pit laps (0-indexed): {pit_laps}")