"""Test multi_channel_distinguishability."""

import pytest

from toqito.channel_metrics import multi_channel_distinguishability
from toqito.channel_ops import kraus_to_choi
from toqito.channels import amplitude_damping, dephasing, phase_damping

# Creating two amplitude damping channels.
amp_damp_1 = kraus_to_choi(amplitude_damping(gamma=0.22))
amp_damp_2 = kraus_to_choi(amplitude_damping(gamma=0.35))

# In Kraus representation.
amp_damp_1_kraus = amplitude_damping(gamma=0.22)
amp_damp_2_kraus = amplitude_damping(gamma=0.35)

# Creating two phase damping channels.
ph_damp_1 = kraus_to_choi(phase_damping(gamma=0.22))
ph_damp_2 = kraus_to_choi(phase_damping(gamma=0.35))


@pytest.mark.parametrize(
    "test_input_1, test_input_2, prior_prob, dim, expected",
    [
        # Distinguishing two identical channels.
        (dephasing(2), dephasing(2), None, [2, 2], 0.5),
        # Distinguishing two amplitude damping channels.
        (amp_damp_1, amp_damp_2, [0.2, 0.8], [2, 2], 0.8),
        # One channel in Kraus and another in Choi representation.
        # (amp_damp_1_kraus, amp_damp_2, [0.2, 0.8], [2, 2], 0.8),
        # Both channels in Kraus representation.
        # (amp_damp_1_kraus, amp_damp_2_kraus, [0.2, 0.8], [2, 2], 0.8),
    ],
)
@pytest.mark.parametrize("primal_dual", ["primal", "dual"])
def test_multi_channel_distinguishability(test_input_1, test_input_2, prior_prob, dim, expected, primal_dual):
    """Test function for n=2 channels to distinguish."""
    calculated_value = multi_channel_distinguishability(
        [test_input_1, test_input_2], prior_prob, dim=dim, primal_dual=primal_dual
    )[0]
    assert pytest.approx(expected, 1e-3) == calculated_value


# @pytest.mark.parametrize(
#     "test_input_1, test_input_2, prior_prob",
#     [
#         # Inconsistent dimensions between two channels.
#         (
#             depolarizing(4),
#             dephasing(2),
#             [0.5, 0.5],
#         ),
#     ],
# )
# @pytest.mark.parametrize(
#     "primal_dual",
#     [
#         "primal",
#         "dual",
#     ],
# )
# def test_state_distinguishability_invalid_channels(test_input_1, test_input_2, prior_prob, primal_dual):
#     """Test function raises error for invalid channel dimensions."""
#     with pytest.raises(
#         ValueError,
#         match="The channels must have the same dimension input and output spaces as each other.",
#     ):
#         multi_channel_distinguishability([test_input_1, test_input_2], prior_prob, primal_dual=primal_dual)
#
#
# @pytest.mark.parametrize(
#     "test_input_1, test_input_2, prior_prob, dim",
#     [
#         (
#             dephasing(2),
#             dephasing(2),
#             [0.5, 0.5],
#             [2, 2],
#         ),
#     ],
# )
# @pytest.mark.parametrize(
#     "primal_dual",
#     [
#         "Random",
#     ],
# )
# def test_state_distinguishability_invalid_primal_dual(test_input_1, test_input_2, prior_prob, dim, primal_dual):
#     """Test function raises error for strategy other than `Bayesian` or `Minimax`."""
#     with pytest.raises(
#         ValueError,
#         match="primal_dual must be either 'primal' or 'dual'",
#     ):
#         multi_channel_distinguishability([test_input_1, test_input_2], prior_prob, dim, primal_dual=primal_dual)
#
#
# @pytest.mark.parametrize(
#     "test_input1, test_input_2, prior_prob, dim, expected_msg",
#     [
#         # Sum of prior probabilities greater than 1.
#         (
#             dephasing(2),
#             dephasing(2),
#             [0.5, 0.9],
#             [2, 2],
#             "Sum of prior probabilities must add up to 1.",
#         ),
#         # Negative terms in prior probabilities
#         (
#             dephasing(2),
#             dephasing(2),
#             [-0.5, 0.5],
#             [2, 2],
#             "probs has negative terms and thus isn't a valid probability distribution.",
#         ),
#         # Length of prior probability list not equal to two.
#         (
#             dephasing(2),
#             dephasing(2),
#             [0.5],
#             [2, 2],
#             "probs must be a probability distribution with as many entries as channels.",
#         ),
#     ],
# )
# def test_bayesian_channel_distinguishability_invalid_inputs(test_input1, test_input_2, prior_prob, dim, expected_msg):
#     """Test function raises error as expected for invalid inputs for bayesian setting."""
#     with pytest.raises(ValueError, match=expected_msg):
#         multi_channel_distinguishability([test_input1, test_input_2], prior_prob, dim)
#
#
# @pytest.mark.parametrize("dim", [2, 3, 5])
# @pytest.mark.parametrize("primal_dual", ["primal", "dual"])
# def test_multi_channel_depolarizing_discrimination(dim, primal_dual):
#     """Test the function on 3 depolarizing channels with different noise parameters."""
#     alpha, beta, gamma = 0.25, 0.5, 0.75
#     depolarizing_alpha = depolarizing(dim, alpha)
#     depolarizing_beta = depolarizing(dim, beta)
#     depolarizing_gamma = depolarizing(dim, gamma)
#
#     assert (
#         pytest.approx(
#             multi_channel_distinguishability(
#                 [depolarizing_alpha, depolarizing_beta, depolarizing_gamma], dim=[dim, dim], primal_dual=primal_dual
#             ),
#             1e-3,
#         )
#         == 1
#     )
#
#
# def test_multi_channel_orthogonal_unitaries_discrimination():
#     """Test the function on orthogonal unitaries."""
#     pass
