"""When to call a truck: a decision rule on top of the forecast and a simulator to price it.

Per route, goods ready for shipment accumulate at the dock (half-hour volumes ``x``). A truck holds
``capacity`` and arrives ``lead`` steps after it is called. A rule calls the next truck once
``backlog + expected inflow over the lead time >= capacity``; with no forecast (inflow = 0) this is
the reactive "call when a load is ready" practice. The simulator replays real volumes and counts
what the business pays for: trucks idling at the dock and full loads waiting for a truck.

The data has no truck sizes, lead times or costs, so they are parameters; results compare rules
under the same assumptions rather than claim absolute savings.
"""

from dataclasses import dataclass

import numpy as np

STEP_H = 0.5  # hours per step


def inflow_forecast(pred: np.ndarray, x0: np.ndarray, k: int) -> np.ndarray:
    """Expected half-hour volume over the next k steps from hourly forecasts.

    pred: (n_o, R, H) forecasts of y[o+j] = x[o+j] + x[o+j-1]; x0: (n_o, R) the known x[o].
    Summing y_1..y_k counts x_1..x_{k-1} twice, x_0 and x_k once, so
    sum x_1..x_k = (sum y_1..y_k + x_k - x_0) / 2 with the unknown x_k taken as y_k / 2.
    """
    s = pred[:, :, :k].sum(axis=2)
    return np.clip((s + 0.5 * pred[:, :, k - 1] - x0) / 2.0, 0, None)


def inflow_actual(x: np.ndarray, origins: np.ndarray, k: int) -> np.ndarray:
    """(n_o, R) realised volume over the k steps after each origin."""
    return np.stack([x[:, o + 1 : o + 1 + k].sum(axis=1) for o in origins])


@dataclass
class DockState:
    backlog: np.ndarray  # (R,) volume waiting at the dock
    on_the_way: np.ndarray  # (R,) trucks called and not yet arrived
    at_dock: np.ndarray  # (R,) trucks at the dock waiting for a full load

    @property
    def inbound(self) -> np.ndarray:
        return self.on_the_way + self.at_dock


@dataclass
class Result:
    trucks: int
    idle_h: float  # trucks waiting at the dock for cargo
    late_h: float  # full loads waiting for a truck
    partial: int  # trucks that left under-filled after the maximum wait
    fill: float  # mean load factor
    peak_backlog: float  # 95th percentile of backlog in truckloads, dock congestion
    state: DockState | None = None

    def per_day(self, days: float) -> dict[str, float]:
        return {"trucks": self.trucks / days, "idle_h": self.idle_h / days, "late_h": self.late_h / days,
                "partial": self.partial / days, "fill": self.fill, "peak_backlog": self.peak_backlog}


NONE = np.iinfo(np.int64).max


def simulate(x: np.ndarray, origins: np.ndarray, expected: np.ndarray, capacity: np.ndarray,
             lead: int, max_wait: int, slots: int = 4, decide_last: bool = True) -> Result:
    """Replay real half-hour volumes ``x`` (R, T) over the contiguous ``origins``.

    expected: (n_o, R) inflow the rule expects over the next ``lead`` steps at each origin
    (zeros = reactive rule). A rule keeps one truck per expected load on the way, up to ``slots``
    trucks per route, so a lead time longer than the fill time is served by overlapping trucks.
    Every route starts with an empty dock and no truck. ``decide_last=False`` stops before the
    calls of the final step, leaving that decision to the live plan.
    """
    R = x.shape[0]
    rows = np.arange(R)
    B = np.zeros(R)
    due = np.full((R, slots), NONE)  # arrival step of each truck on the way
    here = np.full((R, slots), NONE)  # arrival step of each truck at the dock
    ready = np.full((R, slots), -1)  # when the 1st, 2nd, ... waiting load became complete
    trucks = partial = 0
    idle = late = fill_sum = 0.0
    backlog = np.empty((len(origins), R))
    for i, t in enumerate(origins):
        B += x[:, t]
        for k in range(slots):
            land = due[:, k] == t
            if land.any():
                free = np.argmax(here[land] == NONE, axis=1)
                here[rows[land], free] = t
                due[land, k] = NONE
        for k in range(slots):
            m = (B >= (k + 1) * capacity) & (ready[:, k] < 0)
            ready[m, k] = t

        # Trucks load in arrival order while full loads are waiting.
        for _ in range(slots):
            j = np.argmin(here, axis=1)
            arr = here[rows, j]
            go = (arr != NONE) & (B >= capacity)
            if not go.any():
                break
            trucks += int(go.sum())
            fill_sum += float(go.sum())
            idle += float((t - arr[go]).sum())
            # A load that was complete before its truck came waited for it.
            late += float(np.clip(arr[go] - ready[go, 0], 0, None).sum())
            B[go] -= capacity[go]
            here[rows[go], j[go]] = NONE
            ready[go, :-1] = ready[go, 1:]
            ready[go, -1] = -1

        j = np.argmin(here, axis=1)
        arr = here[rows, j]
        give_up = (arr != NONE) & (t - arr >= max_wait)
        if give_up.any():
            trucks += int(give_up.sum())
            partial += int(give_up.sum())
            idle += float((t - arr[give_up]).sum())
            fill_sum += float((B[give_up] / capacity[give_up]).sum())
            B[give_up] = 0.0
            here[rows[give_up], j[give_up]] = NONE
            ready[give_up] = -1

        last = i == len(origins) - 1
        for _ in range(0 if (last and not decide_last) else slots):
            inbound = (due != NONE).sum(axis=1) + (here != NONE).sum(axis=1)
            call = (inbound < slots) & (B + expected[i] >= capacity * (inbound + 1))
            if not call.any():
                break
            k = np.argmax(due[call] == NONE, axis=1)
            due[rows[call], k] = t + lead
        backlog[i] = B / capacity
    state = DockState(B.copy(), (due != NONE).sum(axis=1), (here != NONE).sum(axis=1))
    return Result(trucks, idle * STEP_H, late * STEP_H, partial, fill_sum / max(trucks, 1),
                  float(np.quantile(backlog, 0.95)), state)


def call_plan(backlog: np.ndarray, inflow_q: np.ndarray, capacity: np.ndarray, lead: int,
              inbound: np.ndarray | None = None) -> np.ndarray:
    """Steps from now until the next truck should be called (0 = now), -1 if not within the horizon.

    inflow_q: (R, H) cumulative expected inflow over the next 1..H steps at the chosen risk level.
    The next truck is needed once backlog + inflow covers one more load than the trucks already
    coming; it is called ``lead`` steps before that moment.
    """
    need = capacity * (1 + (np.zeros(len(backlog)) if inbound is None else inbound))
    H = inflow_q.shape[1]
    out = np.full(len(backlog), -1)
    out[backlog >= need] = 0
    for k in range(H - lead + 1):
        hit = (out < 0) & (backlog + inflow_q[:, k + lead - 1] >= need)
        out[hit] = k
    return out
