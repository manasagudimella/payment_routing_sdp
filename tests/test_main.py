import numpy as np
import pytest

from payment_routing_sdp.main import (
    Allocation,
    ApprovalRates,
    Business,
    SingleState,
    State,
    compute_new_state_allocation,
    compute_total_reward,
    get_approval_rates,
    sdp,
    transition,
)


@pytest.mark.parametrize(
    "input, expected",
    [
        [(0.37, 0.365, 0.265), (0.842, 0.679, 0.496)],
        [(0.3833, 0.3782, 0.2385), (0.84, 0.678, 0.495)],
        [(0.433, 0.3335, 0.2335), (0.835, 0.683, 0.495)],
    ],
)
def test_get_approval_rates(
    input,
    expected,
) -> None:
    result = get_approval_rates(input)
    assert result == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize(
    "action, allocation, delta, expected",
    [
        pytest.param(
            1,
            np.array([0.37, 0.365, 0.265]),
            0.1,
            np.array([0.37, 0.365, 0.265]),
            id="action_1",
        ),
        pytest.param(
            2,
            np.array([0.37, 0.365, 0.265]),
            0.1,
            np.array([0.383340136, 0.378159864, 0.2385]),
            id="action_2",
        ),
        pytest.param(
            3,
            np.array([0.37, 0.365, 0.265]),
            0.1,
            np.array([0.391267717, 0.3285, 0.280232283]),
            id="action_3",
        ),
        pytest.param(
            4,
            np.array([0.37, 0.365, 0.265]),
            0.1,
            np.array([0.333, 0.386436508, 0.280563492]),
            id="action_4",
        ),
        pytest.param(
            5,
            np.array([0.37, 0.365, 0.265]),
            0.1,
            np.array([0.33825, 0.4285, 0.23325]),
            id="action_5",
        ),
        pytest.param(
            6,
            np.array([0.37, 0.365, 0.265]),
            0.1,
            np.array([0.33325, 0.32825, 0.3385]),
            id="action_6",
        ),
        pytest.param(
            7,
            np.array([0.37, 0.365, 0.265]),
            0.1,
            np.array([0.433, 0.3335, 0.2335]),
            id="action_7",
        ),
    ],
)
def test_compute_new_state_allocation(
    action: int,
    allocation: Allocation,
    delta: float,
    expected: Allocation,
) -> None:
    result = compute_new_state_allocation(action, allocation, delta)
    assert sum(result) == pytest.approx(1.0)
    assert sum(expected) == pytest.approx(1.0)
    assert result == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("n_transactions", [2400])
@pytest.mark.parametrize(
    "allocation, approval_rates, expected",
    [
        pytest.param(
            np.array([0.37, 0.365, 0.265]),
            np.array([0.842, 0.679, 0.496]),
            1658,
            id="param_1",
        ),
        pytest.param(
            np.array([0.383340136, 0.378159864, 0.2385]),
            np.array([0.84, 0.678, 0.495]),
            1671,
            id="param_2",
        ),
        pytest.param(
            np.array([0.391267717, 0.3285, 0.280232283]),
            np.array([0.84, 0.683, 0.497]),
            1661,
            id="param_3",
        ),
        pytest.param(
            np.array([0.333, 0.386436508, 0.280563492]),
            np.array([0.846, 0.677, 0.497]),
            1639,
            id="param_4",
        ),
        pytest.param(
            np.array([0.33825, 0.4285, 0.23325]),
            np.array([0.845, 0.674, 0.495]),
            1656,
            id="param_5",
        ),
        pytest.param(
            np.array([0.33325, 0.32825, 0.3385]),
            np.array([0.846, 0.683, 0.5]),
            1621,
            id="param_6",
        ),
        pytest.param(
            np.array([0.433, 0.3335, 0.2335]),
            np.array([0.835, 0.683, 0.495]),
            1692,
            id="param_7",
        ),
    ],
)
def test_compute_total_reward(
    n_transactions: int,
    allocation: Allocation,
    approval_rates: ApprovalRates,
    expected: float,
) -> None:
    result = compute_total_reward(n_transactions, allocation, approval_rates)
    assert result == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize(
    "curr_approval_rate, new_approval_rate, exp_state_expanded, exp_state_short",
    [
        pytest.param(
            np.array([0.842, 0.679, 0.496]),
            np.array([0.842, 0.679, 0.496]),
            (1, 1, 1),
            1,
            id="action1",
        ),
        pytest.param(
            np.array([0.842, 0.679, 0.496]),
            np.array([0.84, 0.678, 0.495]),
            (-1, -1, -1),
            8,
            id="action2",
        ),
        pytest.param(
            np.array([0.842, 0.679, 0.496]),
            np.array([0.84, 0.683, 0.497]),
            (-1, 1, 1),
            5,
            id="action3",
        ),
        pytest.param(
            np.array([0.842, 0.679, 0.496]),
            np.array([0.846, 0.677, 0.497]),
            (1, -1, 1),
            3,
            id="action4",
        ),
        pytest.param(
            np.array([0.842, 0.679, 0.496]),
            np.array([0.845, 0.674, 0.495]),
            (1, -1, -1),
            4,
            id="action5",
        ),
        pytest.param(
            np.array([0.842, 0.679, 0.496]),
            np.array([0.846, 0.683, 0.5]),
            (1, 1, 1),
            1,
            id="action6",
        ),
        pytest.param(
            np.array([0.842, 0.679, 0.496]),
            np.array([0.835, 0.683, 0.495]),
            (-1, 1, -1),
            6,
            id="action7",
        ),
    ],
)
def test_state_from_approval_rates(
    curr_approval_rate: ApprovalRates,
    new_approval_rate: ApprovalRates,
    exp_state_expanded: tuple[SingleState, SingleState, SingleState],
    exp_state_short: int,
) -> None:
    result = State.from_approval_rates(curr_approval_rate, new_approval_rate)
    assert result.expanded == exp_state_expanded
    assert result.short == exp_state_short


@pytest.fixture()
def business(
    day: int,
    curr_state: State,
    next_state: State,
    curr_allocation: Allocation,
    next_allocation: Allocation,
    action: int,
    n_transactions: int,
    total_reward: float,
) -> Business:
    return Business(
        day=day,
        prev_state=curr_state,
        curr_state=next_state,
        prev_allocation=curr_allocation,
        curr_allocation=next_allocation,
        action=action,
        n_transactions=n_transactions,
        total_reward=total_reward,
    )


@pytest.mark.parametrize("delta", [0.1])
@pytest.mark.parametrize("n_transactions", [2400])
@pytest.mark.parametrize(
    "curr_allocation",
    [np.r_[0.37, 0.365, 0.265]],
)
@pytest.mark.parametrize(
    "curr_state",
    [State.from_approval_rates(np.r_[0, 0, 0], np.r_[0.842, 0.679, 0.496])],
)
@pytest.mark.parametrize(
    "action, exp_state_expanded, exp_state_short, exp_total_reward",
    [
        pytest.param(1, (1, 1, 1), 1, 1658, id="action_1"),
        pytest.param(2, (-1, -1, -1), 8, 1671, id="action_2"),
        pytest.param(3, (-1, 1, 1), 5, 1661, id="action_3"),
        pytest.param(4, (1, -1, 1), 3, 1639, id="action_4"),
        pytest.param(5, (1, -1, -1), 4, 1656, id="action_5"),
        pytest.param(6, (1, 1, 1), 1, 1621, id="action_6"),
        pytest.param(7, (-1, 1, -1), 6, 1692, id="action_7"),
    ],
)
def test_transition(
    delta: float,
    n_transactions: int,
    curr_allocation: Allocation,
    curr_state: State,
    action: int,
    exp_state_expanded: tuple[SingleState, SingleState, SingleState],
    exp_state_short: int,
    exp_total_reward: float,
) -> None:
    curr_business = Business(
        day=0,
        prev_state=curr_state,
        curr_state=curr_state,
        prev_allocation=curr_allocation,
        curr_allocation=curr_allocation,
        action=0,
        n_transactions=n_transactions,
        total_reward=0,
    )
    next_business = transition(business=curr_business, action=action, delta=delta)
    next_state = next_business.curr_state
    assert next_state.expanded == exp_state_expanded
    assert next_state.short == exp_state_short
    assert next_business.total_reward == pytest.approx(exp_total_reward, rel=1e-3)


@pytest.mark.parametrize(
    "business, exp_states_short, exp_total_rewards",
    [
        pytest.param(
            Business(
                day=0,
                prev_state=State.from_approval_rates(
                    np.r_[0, 0, 0], np.r_[0.842, 0.679, 0.496]
                ),
                curr_state=State.from_approval_rates(
                    np.r_[0, 0, 0], np.r_[0.842, 0.679, 0.496]
                ),
                prev_allocation=np.r_[0.37, 0.365, 0.265],
                curr_allocation=np.r_[0.37, 0.365, 0.265],
                action=1,
                n_transactions=2400,
                total_reward=0,
            ),
            [1, 3, 4, 5, 6, 8],
            [1658, 1639, 1656, 1661, 1692, 1671],
        ),
    ],
)
def test_sdp(
    business: Business, exp_states_short: list[int], exp_total_rewards: list[float]
) -> None:
    next_businesses = sdp([business], 0.1)
    curr_states_short = [business.prev_state.short for business in next_businesses]
    assert curr_states_short == [1, 1, 1, 1, 1, 1]
    states_short = [business.curr_state.short for business in next_businesses]
    total_rewards = [business.total_reward for business in next_businesses]
    assert states_short == exp_states_short
    assert total_rewards == pytest.approx(exp_total_rewards, rel=1e-3)


@pytest.mark.parametrize(
    "businesses",
    [
        pytest.param(
            Business(
                day=0,
                prev_state=State.from_approval_rates(
                    np.r_[0, 0, 0], np.r_[0.842, 0.679, 0.496]
                ),
                curr_state=State.from_approval_rates(
                    np.r_[0, 0, 0], np.r_[0.842, 0.679, 0.496]
                ),
                prev_allocation=np.r_[0.37, 0.365, 0.265],
                curr_allocation=np.r_[0.37, 0.365, 0.265],
                action=1,
                n_transactions=2400,
                total_reward=0,
            ),
        ),
    ],
)
def test_sdp_two_step(businesses: list[Business]) -> None:
    """When I run `sdp` for two steps, for each step the next_state of the old business
    must be the curr_state of the new business.
    """
