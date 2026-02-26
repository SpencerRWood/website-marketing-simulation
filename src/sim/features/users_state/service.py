from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import exp, log


@dataclass(frozen=True)
class UsersSelectionConfig:
    mode: str = "recency_propensity_weighted"
    recency_half_life_hours: float = 18.0
    propensity_weight: float = 0.5
    recency_weight: float = 0.5


@dataclass(frozen=True)
class PropensityInitConfig:
    dist: str = "uniform"  # "uniform" | "beta"
    alpha: float = 2.0
    beta: float = 6.0


@dataclass(frozen=True)
class DiscoveryModeConfig:
    enabled: bool = True
    graduation_sessions: int = 2
    dropoff_multiplier: float = 1.25
    conversion_logit_shift: float = -0.8


@dataclass(frozen=True)
class UsersConfig:
    new_user_share: float = 0.6
    selection: UsersSelectionConfig = UsersSelectionConfig()
    propensity_init: PropensityInitConfig = PropensityInitConfig()
    discovery_mode: DiscoveryModeConfig = DiscoveryModeConfig()


@dataclass
class UserState:
    user_id: str
    created_ts_utc: datetime
    last_seen_ts_utc: datetime
    sessions_count: int
    propensity: float
    discovery_mode: bool
    discovery_sessions_count: int

    # Convenience fields (copied from config for downstream use)
    discovery_dropoff_multiplier: float
    discovery_conversion_logit_shift: float


class UsersStateService:
    """
    Hot storage for users. Deterministic selection via caller-provided RNG.
    - Users persist in-memory for entire run.
    - New vs existing chosen per users.new_user_share.
    - Existing selection uses recency+propensity weighted sampling.
    - Discovery mode graduation after N sessions.
    """

    def __init__(self, cfg: UsersConfig):
        self.cfg = cfg
        self.users: dict[str, UserState] = {}
        self._next_user_seq: int = 0

    # ----------------------------
    # Public API
    # ----------------------------
    def get_or_create_user_for_intent(
        self,
        *,
        now_utc: datetime,
        rng,
    ) -> tuple[UserState, bool]:
        """
        Returns (user, is_new_user).

        Rules:
        - If no users exist, always create.
        - Else: create new w.p. new_user_share, otherwise select existing.
        """
        now_utc = _ensure_utc(now_utc)

        if not self.users:
            u = self._create_user(now_utc=now_utc, rng=rng)
            return u, True

        p_new = float(self.cfg.new_user_share)
        if rng.random() < p_new:
            u = self._create_user(now_utc=now_utc, rng=rng)
            return u, True

        u = self.select_existing_user(now_utc=now_utc, rng=rng)
        # Defensive: if selection fails (should not), fall back to create.
        if u is None:
            u = self._create_user(now_utc=now_utc, rng=rng)
            return u, True
        return u, False

    def select_existing_user(self, *, now_utc: datetime, rng) -> UserState | None:
        """
        Selects an existing user via configured weighting.
        """
        now_utc = _ensure_utc(now_utc)
        if not self.users:
            return None

        mode = (self.cfg.selection.mode or "").strip().lower()
        if mode != "recency_propensity_weighted":
            raise ValueError(f"Unsupported users.selection.mode={self.cfg.selection.mode!r}")

        candidates = list(self.users.values())
        weights = [self._weight(u, now_utc=now_utc) for u in candidates]

        # If all weights are 0, fall back to uniform.
        if sum(weights) <= 0:
            idx = int(rng.random() * len(candidates))
            return candidates[idx]

        return rng.choices(candidates, weights=weights, k=1)[0]

    def mark_session_end(self, *, user_id: str, now_utc: datetime) -> None:
        """
        Update last_seen + session counters; handle discovery graduation.
        """
        now_utc = _ensure_utc(now_utc)
        u = self.users.get(user_id)
        if u is None:
            raise KeyError(f"Unknown user_id={user_id}")

        u.last_seen_ts_utc = now_utc
        u.sessions_count += 1

        if u.discovery_mode:
            u.discovery_sessions_count += 1
            grad_n = int(self.cfg.discovery_mode.graduation_sessions)
            if grad_n > 0 and u.discovery_sessions_count >= grad_n:
                u.discovery_mode = False

    def get_user(self, user_id: str) -> UserState | None:
        return self.users.get(user_id)

    def all_users(self) -> Iterable[UserState]:
        return self.users.values()

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _create_user(self, *, now_utc: datetime, rng) -> UserState:
        self._next_user_seq += 1
        user_id = f"u_{self._next_user_seq:010d}"

        propensity = self._init_propensity(rng)

        discovery_enabled = bool(self.cfg.discovery_mode.enabled)
        u = UserState(
            user_id=user_id,
            created_ts_utc=now_utc,
            last_seen_ts_utc=now_utc,
            sessions_count=0,
            propensity=float(propensity),
            discovery_mode=discovery_enabled,
            discovery_sessions_count=0,
            discovery_dropoff_multiplier=float(self.cfg.discovery_mode.dropoff_multiplier),
            discovery_conversion_logit_shift=float(self.cfg.discovery_mode.conversion_logit_shift),
        )
        self.users[user_id] = u
        return u

    def _init_propensity(self, rng) -> float:
        cfg = self.cfg.propensity_init
        dist = (cfg.dist or "uniform").strip().lower()

        if dist == "uniform":
            return float(rng.random())

        if dist == "beta":
            # We cannot assume numpy; implement Beta via Gamma(a,1)/(Gamma(a,1)+Gamma(b,1))
            a = float(cfg.alpha)
            b = float(cfg.beta)
            if a <= 0 or b <= 0:
                # fall back to uniform if misconfigured
                return float(rng.random())
            x = _gamma_sample(shape=a, rng=rng)
            y = _gamma_sample(shape=b, rng=rng)
            if x + y <= 0:
                return float(rng.random())
            return float(x / (x + y))

        # Unknown dist -> uniform
        return float(rng.random())

    def _weight(self, u: UserState, *, now_utc: datetime) -> float:
        sel = self.cfg.selection

        # Recency decay using half-life
        half_life_h = max(float(sel.recency_half_life_hours), 1e-9)
        age_s = max(0.0, (now_utc - u.last_seen_ts_utc).total_seconds())
        age_h = age_s / 3600.0

        # exp(-ln2 * age/half_life) => 1 at age=0, 0.5 at age=half_life
        recency_score = exp(-log(2.0) * (age_h / half_life_h))

        propensity_score = max(0.0, min(1.0, float(u.propensity)))

        w = (
            float(sel.recency_weight) * recency_score
            + float(sel.propensity_weight) * propensity_score
        )
        return max(0.0, w)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _gamma_sample(*, shape: float, rng) -> float:
    """
    Marsaglia and Tsang method for Gamma(shape, 1).
    shape > 0. No numpy dependency.
    """
    # Special-case small shapes using boosting: Gamma(k) = Gamma(k+1) * U^(1/k)
    if shape < 1.0:
        u = rng.random()
        return _gamma_sample(shape=shape + 1.0, rng=rng) * (u ** (1.0 / shape))

    d = shape - 1.0 / 3.0
    c = 1.0 / (3.0 * d) ** 0.5

    while True:
        x = _std_normal(rng)
        v = 1.0 + c * x
        if v <= 0:
            continue
        v = v**3
        u = rng.random()
        if u < 1.0 - 0.0331 * (x**4):
            return d * v
        if log(u) < 0.5 * x * x + d * (1.0 - v + log(v)):
            return d * v


def _std_normal(rng) -> float:
    """
    Box-Muller transform using rng.random().
    """
    # Avoid log(0)
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    from math import cos, pi, sqrt
    from math import log as mlog

    return sqrt(-2.0 * mlog(u1)) * cos(2.0 * pi * u2)
