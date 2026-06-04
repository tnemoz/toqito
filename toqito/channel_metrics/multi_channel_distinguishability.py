"""Computes the maximum probability of distinguishing two or more quantum channels."""

from typing import Any

import numpy as np
import picos as pc

from toqito.channel_ops import kraus_to_choi
from toqito.channel_props.channel_dim import channel_dim


def multi_channel_distinguishability(
    channels: list[np.ndarray | list[np.ndarray] | list[list[np.ndarray]]],
    probs: list[float] | None = None,
    dim: int | list[int] | np.ndarray | None = None,
    solver: str = "cvxopt",
    primal_dual: str = "dual",
    **kwargs: Any,
) -> tuple[float | np.floating, list[pc.HermitianVariable] | list[np.ndarray] | tuple[pc.SymmetricVariable]]:
    r"""Compute the optimal probability of distinguishing two or more quantum channels.

    Only Bayesian discrimination of two or more quantum channels is implemented.

    Channels to be distinguished should have a given a priori probability distribution.
    The task of discriminating channels can be solved by the following SDP:

    \[
        \begin{align*}
            \text{maximize:} \quad & \sum_{i=1}^n p_i\mathrm{Tr}\left[\mu_iJ\left(\Phi_i\right)\right] \\
            \text{subject to:} \quad & \mu_1 + \ldots + \mu_n = \rho\otimes\mathbb{I}_{\text{out}},\\
                                     & \mu_1, \ldots, \mu_n \succeq 0,\\
                                     & \rho\succeq0,\\
                                     & \mathrm{Tr}[\rho]=1
        \end{align*}
    \]

    with \(\left\{p_i\right\}_i\) being the prior probabilities of the channels \(\left\{\Phi_i\right\}_i\) and
    \(J(\Psi)\) being the CHoi matrix of the quantum channel \(\Psi\). The dual problem is then given by

    \[
        \begin{align*}
            \text{minimize:} \quad & \lambda \\
            \text{subject to:} \quad & \forall i,\lambda\mathbb{I}_{\text{in}} \succeq
                                       \mathrm{Tr}_{\text{out}}\left[p_iJ\left(\Phi_i\right)-W\right],\\
                                     & \forall i, p_iJ\left(\Phi_i\right)\succeq W.
        \end{align*}
    \]

    Args:
        channels: A list of superoperators. Each superoperator should be provided either as a Choi matrix,
             or as a (1d or 2d) list of numpy arrays whose entries are its Kraus operators.
        probs: Prior probabilities of the channels. If no probabilities are provided, a uniform
            probability distribution is assumed.
        dim: Input and output dimensions of the channels.
        solver: Optimization option for `picos` solver. Default option is `solver="cvxopt"`.
        primal_dual: Option for the optimization problem. Default option is `primal_dual="dual"`.
        kwargs: Additional arguments to pass to picos' solve method.

    Returns:
        The optimal probability with which Bob can guess the channel he was given from `channels` along with the optimal
        1-tester.

    Raises:
        ValueError: If channels have different input or output dimensions.
        ValueError: If prior probabilities do not add up to 1.
        ValueError: If number of prior probabilities is not equal to the number of channels.

    Examples:
        Optimal probability of distinguishing two amplitude damping channels in the Bayesian setting:

        ```python exec="1" source="above" result="text"
        from toqito.channels import amplitude_damping
        from toqito.channel_ops import kraus_to_choi
        from toqito.channel_metrics import multi_channel_distinguishability
        # Define three amplitude damping channels with gamma=0.25, gamma=0.5 and gamma=0.75
        choi_ch_1 = kraus_to_choi(amplitude_damping(gamma=0.25))
        choi_ch_2 = kraus_to_choi(amplitude_damping(gamma=0.5))
        choi_ch_3 = kraus_to_choi(amplitude_damping(gamma=0.5))

        print(multi_channel_distinguishability([choi_ch_1, choi_ch_2, choi_ch_3]))
        ```

    """
    # Get the input, output and environment dimensions of each channel.
    dim_in, dim_out, dim_env = channel_dim(channels[0], dim=dim)

    for channel in channels[1:]:
        test_dim_in, test_dim_out, test_dim_env = channel_dim(channel, dim=dim)

        if not np.array_equal(np.array([dim_in, dim_out]), np.array([test_dim_in, test_dim_out])):
            raise ValueError("The channels must have the same dimension input and output spaces as each other.")

    # If a channel is provided as a list, we assume this is a list
    # of Kraus operators. We convert to choi matrices if not provided as choi matrix.

    choi_channels = []

    for phi in channels:
        if isinstance(phi, list):
            choi_channels.append(kraus_to_choi(phi))
        else:
            choi_channels.append(phi)

    if probs is None:
        probs = [1 / len(channels) for _ in channels]

    for index, p in enumerate(probs):
        choi_channels[index] = choi_channels[index] * p

    if len(probs) != len(channels):
        raise ValueError("probs must be a probability distribution with as many entries as channels.")

    if min(probs) < 0:
        raise ValueError("probs has negative terms and thus isn't a valid probability distribution.")

    if abs(sum(probs) - 1) != 0:
        raise ValueError("Sum of prior probabilities must add up to 1.")

    if primal_dual not in ["primal", "dual"]:
        raise ValueError(f"primal_dual must be either 'primal' or 'dual' but {primal_dual} was found.")

    if primal_dual == "primal":
        return _bayesian_primal(choi_channels, dim_in[0], dim_out[0], solver=solver, **kwargs)

    return _bayesian_dual(choi_channels, dim_in[0], dim_out[0], solver=solver, **kwargs)


def _bayesian_dual(
    scaled_channels: list[np.ndarray],
    dim_in: int,
    dim_out: int,
    solver: str,
    **kwargs,
) -> tuple[float, list[pc.HermitianVariable]]:
    """Solve the dual problem for bayesian quantum channel distinguishability SDP."""
    problem = pc.Problem()

    lam = pc.RealVariable("lambda")
    W = pc.HermitianVariable("W", (dim_in * dim_out, dim_in * dim_out))

    problem.add_list_of_constraints(W << scaled_choi for scaled_choi in scaled_channels)
    problem.add_constraint(pc.partial_trace(W, 1, [int(dim_in), int(dim_out)]) >> lam * pc.I(dim_in))

    problem.set_objective("min", lam)

    problem.solve(solver=solver, **kwargs)

    tester = [problem.get_constraint(i).dual for i in range(len(scaled_channels))]

    return problem.value, tester


def _bayesian_primal(
    scaled_channels: list[np.ndarray],
    dim_in: int,
    dim_out: int,
    solver: str,
    **kwargs,
) -> tuple[float, list[pc.HermitianVariable]]:
    """Solve the primal problem for bayesian quantum channel distinguishability SDP."""
    problem = pc.Problem()

    tester = [
        pc.HermitianVariable(f"mu_{i}", (dim_in * dim_out, dim_in * dim_out)) for i in range(len(scaled_channels))
    ]
    rho = pc.HermitianVariable("rho", (dim_in, dim_in))

    problem.add_list_of_constraints(mu_i >> 0 for mu_i in tester)
    problem.add_constraint(pc.sum(tester) == pc.I(dim_out) @ rho.T)
    problem.add_constraint(rho >> 0)
    problem.add_constraint(pc.trace(rho) == 1)

    problem.set_objective(
        "max", pc.trace(pc.sum(mu_i * scaled_choi for mu_i, scaled_choi in zip(tester, scaled_channels)))
    )

    problem.solve(solver=solver, **kwargs)

    return problem.value / dim_in, tester
