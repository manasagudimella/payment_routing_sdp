import dataclasses
import itertools
import math
import typing as t

import numpy as np
import rich
from rich.table import Table

DELTA = 0.1  # The amount by which the allocation changes in each action

# Type alias for a 3-element numpy array of floats
Allocation: t.TypeAlias = np.ndarray[t.Literal[3], np.dtype[np.float64]]
ApprovalRates: t.TypeAlias = np.ndarray[t.Literal[3], np.dtype[np.float64]]
SingleState: t.TypeAlias = t.Literal[1, -1]


@dataclasses.dataclass
class State:
    short: int
    expanded: tuple[SingleState, SingleState, SingleState]
    approval_rates: ApprovalRates

    STATE_MAP: t.ClassVar[dict[tuple[SingleState, SingleState, SingleState], int]] = {
        (1, 1, 1): 1,
        (1, 1, -1): 2,
        (1, -1, 1): 3,
        (1, -1, -1): 4,
        (-1, 1, 1): 5,
        (-1, 1, -1): 6,
        (-1, -1, 1): 7,
        (-1, -1, -1): 8,
    }
    REVERSE_STATE_MAP: t.ClassVar[
        dict[int, tuple[SingleState, SingleState, SingleState]]
    ] = {v: k for k, v in STATE_MAP.items()}

    @classmethod
    def from_short(cls, short: int, approval_rates: ApprovalRates) -> "State":
        return cls(short, cls.REVERSE_STATE_MAP[short], approval_rates=approval_rates)

    @classmethod
    def from_expanded(
        cls,
        expanded: tuple[SingleState, SingleState, SingleState],
        approval_rates: ApprovalRates,
    ) -> "State":
        return cls(cls.STATE_MAP[expanded], expanded, approval_rates=approval_rates)

    @classmethod
    def from_approval_rates(
        cls, prev_approval_rates: ApprovalRates, curr_approval_rates: ApprovalRates
    ) -> "State":
        assert len(prev_approval_rates) == len(curr_approval_rates) == 3
        return cls.from_expanded(
            tuple(  # type: ignore
                1 if qi >= pi else -1
                for pi, qi in zip(prev_approval_rates, curr_approval_rates)
            ),
            approval_rates=curr_approval_rates,
        )


@dataclasses.dataclass
class Business:
    day: int
    prev_state: State | None
    curr_state: State
    prev_allocation: Allocation | None
    curr_allocation: Allocation
    action: int
    n_transactions: int
    curr_reward: float
    prev_business: "Business | None" = None

    def __rich_repr__(self):
        table = Table(title="Business")
        table.add_column("Attribute")
        table.add_column("Value")
        table.add_row("Day", f"{self.day}")
        table.add_row(
            "Prev State", f"{self.prev_state.short if self.prev_state else None}"
        )
        table.add_row("Curr State", f"{self.curr_state.short}")
        if self.prev_allocation is not None:
            table.add_row(
                "Prev Allocation",
                ", ".join([f"{x:.3f}" for x in self.prev_allocation]),
            )
        table.add_row(
            "Curr Allocation", ", ".join([f"{x:.3f}" for x in self.curr_allocation])
        )
        table.add_row("Action", f"{self.action}")
        table.add_row("N Transactions", f"{self.n_transactions}")
        table.add_row("Curr Reward", f"{self.curr_reward:.3f}")
        table.add_row("Total Reward", f"{self.total_reward:.3f}")
        return table

    @property
    def total_reward(self):
        return self.curr_reward + (
            self.prev_business.total_reward if self.prev_business else 0
        )


def get_approval_rates(allocation: Allocation) -> ApprovalRates:
    """Calculate the approval rates for a given allocation of providers."""

    # Define the coefficients for the approval rate function per provider
    provider0 = (2.0, -1.0, 0.3)
    provider1 = (1.0, -0.9, 0.6)
    provider2 = (0.0, -0.3, 0.9)

    def approval_rate_for_provider(
        allocation: float, alpha: float, beta: float, gamma: float
    ) -> float:
        exp_term = np.exp(alpha + beta * allocation + gamma * allocation**2)
        return np.round(exp_term / (1 + exp_term), 3)

    return np.array(
        [
            approval_rate_for_provider(allocation[0], *provider0),
            approval_rate_for_provider(allocation[1], *provider1),
            approval_rate_for_provider(allocation[2], *provider2),
        ]
    )


def compute_new_state_allocation(action: int, allocation: Allocation, delta: float):
    # sourcery skip: extract-duplicate-method, move-assign-in-block, switch
    allocation = allocation.copy()
    if action == 1:
        return allocation
    elif action == 2:  # a=2 means (+,+,-)
        amount_to_adjust = allocation[2] * delta
        ratio = allocation[1] / (allocation[0] + allocation[1])
        allocation[2] = 0.9 * allocation[2]
        allocation[1] = allocation[1] + amount_to_adjust * ratio
        allocation[0] = 1 - allocation[1] - allocation[2]
        return allocation
    elif action == 3:  # a=3 means (+,-,+)
        amount_to_adjust = allocation[1] * delta
        ratio = allocation[2] / (allocation[0] + allocation[2])
        allocation[1] = 0.9 * allocation[1]
        allocation[2] = allocation[2] + amount_to_adjust * ratio
        allocation[0] = 1 - allocation[1] - allocation[2]
        return allocation
    elif action == 4:  # a=4 means (-,+,+)
        amount_to_adjust = allocation[0] * delta
        ratio = allocation[2] / (allocation[1] + allocation[2])
        allocation[0] = allocation[0] - amount_to_adjust
        allocation[2] = allocation[2] + amount_to_adjust * ratio
        allocation[1] = 1 - allocation[0] - allocation[2]
        return allocation
    elif action == 5:  # a=5 means (-,+,-)
        amount_to_adjust = (allocation[0] + allocation[2]) * delta
        ratio = 0.5
        allocation[1] = allocation[1] + amount_to_adjust
        allocation[2] = allocation[2] - amount_to_adjust * ratio
        allocation[0] = 1 - allocation[1] - allocation[2]
        return allocation
    elif action == 6:  # a=6 means (-,-,+)
        amount_to_adjust = (allocation[0] + allocation[1]) * delta
        ratio = 0.5
        allocation[2] = allocation[2] + amount_to_adjust
        allocation[0] = allocation[0] - amount_to_adjust * ratio
        allocation[1] = 1 - allocation[0] - allocation[2]
        return allocation
    elif action == 7:  # a=7 means (+,-,-)
        amount_to_adjust = (allocation[1] + allocation[2]) * delta / 2
        allocation[1] = allocation[1] - amount_to_adjust
        allocation[2] = allocation[2] - amount_to_adjust
        allocation[0] = 1 - allocation[1] - allocation[2]
        return allocation
    else:
        raise ValueError(f"Invalid action: {action}")


def transition(
    business: Business,
    action: int,
    delta: float,
    n_transactions: int | None = None,
    horizon: int = 1,
) -> Business:
    n_transactions = (
        business.n_transactions if n_transactions is None else n_transactions
    )
    # First day take the action and compute the reward
    final_allocation = new_allocation = compute_new_state_allocation(
        action, business.curr_allocation, delta
    )
    new_approval_rates = get_approval_rates(new_allocation)
    final_state = new_state = State.from_approval_rates(
        business.curr_state.approval_rates, new_approval_rates
    )
    curr_reward = compute_total_reward(
        n_transactions, new_allocation, new_approval_rates
    )
    for _ in range(1, horizon):
        # For all remaining days, don't take any action
        new_allocation = compute_new_state_allocation(1, new_allocation, delta)
        new_approval_rates = get_approval_rates(new_allocation)
        new_state = State.from_approval_rates(
            new_state.approval_rates, new_approval_rates
        )
        curr_reward = curr_reward + compute_total_reward(
            n_transactions, new_allocation, new_approval_rates
        )
    return Business(
        day=business.day + horizon,
        prev_state=business.curr_state,
        curr_state=final_state,
        prev_allocation=business.curr_allocation,
        curr_allocation=final_allocation,
        action=action,
        n_transactions=n_transactions,
        curr_reward=curr_reward,
        prev_business=business,
    )


def compute_total_reward(
    n_transactions: int, allocation: Allocation, approval_rates: ApprovalRates
):
    return n_transactions * sum(allocation * approval_rates)


def sdp(
    businesses: list[Business],
    delta: float,
    n_transactions: int | None = None,
    debug: bool = False,
    horizon: int = 1,
) -> list[Business]:
    # sourcery skip: use-itertools-product
    business_day = businesses[0].day
    assert all(business.day == business_day for business in businesses)
    all_states = [x.curr_state.short for x in businesses]
    assert len(all_states) <= len(State.STATE_MAP)
    next_businesses = []
    for business in businesses:
        for action in range(1, 8):
            next_business = transition(
                business, action, delta, n_transactions, horizon=horizon
            )
            next_businesses.append(next_business)
    grouped_by_state = itertools.groupby(
        sorted(next_businesses, key=lambda b: b.curr_state.short),
        key=lambda b: b.curr_state.short,
    )
    grouped_by_state = [(key, list(group)) for key, group in grouped_by_state]
    display_debug_info(debug, business_day, grouped_by_state)
    result = {
        key: max(group, key=lambda b: b.total_reward) for key, group in grouped_by_state
    }
    return list(result.values())


def display_debug_info(
    debug: bool, business_day: int, grouped_by_state: list[t.Tuple[int, list[Business]]]
):
    table = Table(title=f"Debug Day: {business_day}")
    table.add_column("Group State")
    table.add_column("Curr State")
    table.add_column("Action")
    table.add_column("Prev State")
    table.add_column("Curr Allocation")
    table.add_column("Curr Approval Rates")
    table.add_column("Curr Reward")
    table.add_column("Total Reward")
    for state, group in grouped_by_state:
        max_reward = max(b.curr_reward for b in group)
        for business in group:
            style = (
                "bold green" if math.isclose(business.curr_reward, max_reward) else ""
            )
            table.add_row(
                f"{state}",
                f"{business.curr_state.short}",
                f"{business.action}",
                f"{business.prev_state.short}",  # type: ignore
                ", ".join([f"{x:.3f}" for x in business.curr_allocation]),
                ", ".join([f"{x:.3f}" for x in business.curr_state.approval_rates]),
                f"{business.curr_reward:.3f}",
                f"{business.total_reward:.3f}",
                style=style,
            )
        table.add_row(
            "---",
            "---",
            "---",
            "---",
            "---",
            "---",
            "---",
        )
    if debug:
        rich.print(table)


def main(n_days: int, seed: int = 42, horizon: int = 1, debug: bool = False) -> float:
    rng = np.random.default_rng(seed)
    assert n_days % horizon == 0, "n_days must be divisible by horizon"
    businesses = [
        Business(
            day=0,
            prev_state=None,
            curr_state=State.from_approval_rates(
                np.r_[0, 0, 0], np.r_[0.842, 0.679, 0.496]
            ),
            prev_allocation=np.r_[0.37, 0.365, 0.265],
            curr_allocation=np.r_[0.37, 0.365, 0.265],
            action=1,
            n_transactions=2400,
            curr_reward=0,
        )
    ]
    n_transactions = 2400
    print("Day 0")
    print(f"Curr Allocation: {businesses[0].curr_allocation}")
    print(f"Curr Approval Rates: {businesses[0].curr_state.approval_rates}")
    print("N transactions", businesses[0].n_transactions)
    print("Current state", businesses[0].curr_state.short)
    for day in range(n_days // horizon):
        businesses = sdp(
            businesses,
            DELTA,
            n_transactions=n_transactions,
            debug=debug,
            horizon=horizon,
        )
        max_reward = max(business.total_reward for business in businesses)
        table = Table(title=f"Day: {(day + 1) * horizon}")
        table.add_column("Curr State")
        table.add_column("Action")
        table.add_column("Prev State")
        table.add_column("Curr Allocation")
        table.add_column("Curr Approval Rates")
        table.add_column("Curr Reward")
        table.add_column("Total Reward")
        for business in businesses:
            style = (
                "bold green" if math.isclose(business.total_reward, max_reward) else ""
            )
            table.add_row(
                f"{business.curr_state.short}",
                f"{business.action}",
                f"{business.prev_state.short}",  # type: ignore
                ", ".join([f"{x:.3f}" for x in business.curr_allocation]),
                ", ".join([f"{x:.3f}" for x in business.curr_state.approval_rates]),
                f"{business.curr_reward:.3f}",
                f"{business.total_reward:.3f}",
                style=style,
            )
        rich.print(table)
        n_transactions = 2400  # rng.integers(1000, 3000)
        # n_transactions = rng.integers(1000, 3000)
    best_business = max(businesses, key=lambda b: b.total_reward)
    print("Business Trace")
    best_reward = best_business.total_reward
    trace = [best_business]
    while best_business:
        best_business = best_business.prev_business
        if best_business:
            trace.append(best_business)
    for business in reversed(trace):
        rich.print(business.__rich_repr__())
    return best_reward


if __name__ == "__main__":
    horizon = int(input("Enter the horizon: "))
    main(250, horizon=horizon)
