"""
hmm.py
------
Hidden Markov Model — implemented from scratch using NumPy.

Hidden States:
    0 = FRESH      (tyre performing at peak)
    1 = DEGRADING  (lap times rising, wear increasing)
    2 = CRITICAL   (pit stop imminent)

Observations: continuous (Gaussian emissions per state)

Math reference:
    Rabiner (1989) "A Tutorial on Hidden Markov Models"
"""

import numpy as np
from scipy.stats import multivariate_normal


STATE_NAMES = {0: 'FRESH', 1: 'DEGRADING', 2: 'CRITICAL'}
N_STATES = 3


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
        Initialize an ordered, left-to-right model using normalized tyre age.
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
            (np.atleast_2d(np.cov(chunk.T)) if len(chunk) > 1 else np.eye(n_features))
            + np.eye(n_features) * 1e-3
            for chunk in chunks
        ])

    # ------------------------------------------------------------------
    # Emission probability
    # ------------------------------------------------------------------

    def _emission_prob(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute B[j, t] = P(o_t | state=j) for all states and timesteps.

        Uses multivariate Gaussian:
            P(o | μ, Σ) = (2π)^(-d/2) |Σ|^(-1/2) exp(-0.5 (o-μ)^T Σ^{-1} (o-μ))

        Args:
            obs: shape (T, n_features)
        Returns:
            B: shape (T, n_states)  — emission probs at each timestep
        """
        T = obs.shape[0]
        B = np.zeros((T, self.n_states))
        for j in range(self.n_states):
            B[:, j] = multivariate_normal.pdf(obs, mean=self.mu[j], cov=self.sigma[j])
        # Avoid zero probabilities (numerical safety)
        B = np.clip(B, 1e-300, None)
        return B

    # ------------------------------------------------------------------
    # Forward Algorithm  O(T * N^2)
    # ------------------------------------------------------------------

    def _forward(self, obs: np.ndarray, B: np.ndarray) -> tuple:
        """
        Forward algorithm: compute α_t(i) = P(o_1,...,o_t, q_t=i | λ)

        Recursion:
            α_1(i) = π_i * B_i(o_1)
            α_t(j) = [Σ_i α_{t-1}(i) * A_{ij}] * B_j(o_t)

        Uses log-scaling to prevent underflow.

        Returns:
            alpha : shape (T, N)
            log_likelihood : float
        """
        T = obs.shape[0]
        N = self.n_states
        alpha = np.zeros((T, N))
        scale = np.zeros(T)  # scaling factors to prevent underflow

        # Initialization
        alpha[0] = self.pi * B[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0] + 1e-300

        # Recursion
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * B[t]
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t] + 1e-300

        log_likelihood = np.sum(np.log(scale + 1e-300))
        return alpha, log_likelihood

    # ------------------------------------------------------------------
    # Backward Algorithm  O(T * N^2)
    # ------------------------------------------------------------------

    def _backward(self, obs: np.ndarray, B: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Backward algorithm: compute β_t(i) = P(o_{t+1},...,o_T | q_t=i, λ)

        Recursion:
            β_T(i) = 1
            β_t(i) = Σ_j A_{ij} * B_j(o_{t+1}) * β_{t+1}(j)

        Returns:
            beta : shape (T, N)
        """
        T = obs.shape[0]
        N = self.n_states
        beta = np.zeros((T, N))
        beta[T-1] = 1.0

        for t in range(T-2, -1, -1):
            beta[t] = (self.A * B[t+1] * beta[t+1]).sum(axis=1)
            beta[t] /= scale[t+1] + 1e-300

        return beta

    def filter(self, obs: np.ndarray) -> np.ndarray:
        """Return P(state_t | observations through t), without future observations."""
        if obs.ndim != 2 or len(obs) == 0:
            raise ValueError('Filtering requires a non-empty 2D observation sequence.')
        probabilities, _ = self._forward(obs, self._emission_prob(obs))
        return probabilities

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
        B = self._emission_prob(obs)

        # Use log probabilities to avoid underflow
        log_A = np.where(self.A > 0, np.log(np.maximum(self.A, 1e-300)), -np.inf)
        log_pi = np.where(self.pi > 0, np.log(np.maximum(self.pi, 1e-300)), -np.inf)
        log_B = np.log(B + 1e-300)

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

        all_obs = np.vstack(sequences)
        self._initialize_params(all_obs)

        n_features = all_obs.shape[1]
        prev_ll = -np.inf

        print(f"Training HMM ({self.n_states} states, {n_features} features)...")

        for iteration in range(self.n_iter):
            # Accumulators for M-step (summed across all sequences)
            sum_xi = np.zeros((self.n_states, self.n_states))
            sum_gamma = np.zeros(self.n_states)
            sum_gamma_obs = np.zeros((self.n_states, n_features))
            sum_gamma_obs_outer = np.zeros((self.n_states, n_features, n_features))
            total_ll = 0.0

            # ---- E-step ----
            for obs in sequences:
                T = obs.shape[0]
                B = self._emission_prob(obs)

                alpha, ll = self._forward(obs, B)
                total_ll += ll

                # Recompute scale for backward pass
                _, scale = self._forward_with_scale(obs, B)
                beta = self._backward(obs, B, scale)

                # γ_t(i) = α_t(i) * β_t(i) / Σ_j α_t(j)*β_t(j)
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300  # shape (T, N)

                # ξ_t(i,j) = α_t(i) * A_ij * B_j(o_{t+1}) * β_{t+1}(j)
                xi = np.zeros((T-1, self.n_states, self.n_states))
                for t in range(T-1):
                    xi[t] = (alpha[t][:, None] * self.A * B[t+1] * beta[t+1])
                    xi[t] /= xi[t].sum() + 1e-300

                # Accumulate
                sum_xi += xi.sum(axis=0)
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
                self.mu[j] = sum_gamma_obs[j] / (sum_gamma[j] + 1e-300)
                second_moment = sum_gamma_obs_outer[j] / (sum_gamma[j] + 1e-300)
                self.sigma[j] = second_moment - np.outer(self.mu[j], self.mu[j])
                self.sigma[j] += np.eye(n_features) * 1e-3  # Regularize

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

    def _forward_with_scale(self, obs: np.ndarray, B: np.ndarray) -> tuple:
        """Forward pass returning scale factors (needed for backward pass)."""
        T = obs.shape[0]
        N = self.n_states
        alpha = np.zeros((T, N))
        scale = np.zeros(T)

        alpha[0] = self.pi * B[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0] + 1e-300

        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * B[t]
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t] + 1e-300

        return alpha, scale

    def decode(self, obs: np.ndarray) -> list:
        """
        Return named causal, forward-filtered state predictions.

        Returns list of state names e.g. ['FRESH', 'FRESH', 'DEGRADING', 'CRITICAL']
        """
        state_ids = self.filter(obs).argmax(axis=1)
        return [STATE_NAMES[s] for s in state_ids]

    def predict_pit_laps(self, obs: np.ndarray, threshold_state: int = 2) -> list:
        """
        Predict pit stop laps: laps where state transitions INTO CRITICAL (state 2).

        Args:
            obs             : observation sequence shape (T, n_features)
            threshold_state : state index considered "pit now" (default 2 = CRITICAL)
        Returns:
            list of lap indices where a pit stop is predicted
        """
        states = self.filter(obs).argmax(axis=1)
        pit_laps = []
        for t in range(1, len(states)):
            if states[t] == threshold_state and states[t-1] != threshold_state:
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