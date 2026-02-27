from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import simpy

from sim.features.session_intent.types import SessionIntent

from .types import EventSink, IdsLike, IntentResolverConfig, RngLike, SessionRunner, UsersStateLike


@dataclass(slots=True)
class IntentResolverService:
    """
    Consumes SessionIntent messages and resolves them into (user_id, session_id),
    emitting events and spawning a downstream session runner process.

    Responsibilities:
      - Delegate user creation/selection to UsersStateService (hot storage)
      - Generate session IDs (IdsLike)
      - Emit events (EventSink)
      - Spawn downstream session process (SessionRunner)
    """

    env: simpy.Environment
    cfg: IntentResolverConfig

    ids: IdsLike
    rng: RngLike

    # Upstream: expects SessionIntentService-like with .get()
    intents: Any

    # Hot state (UsersStateService-compatible)
    users: UsersStateLike

    # Downstream
    session_runner: SessionRunner

    # Events
    sink: EventSink

    def start(self) -> simpy.events.Process:
        return self.env.process(self._run_loop())

    def _now_ts_utc(self, intent: SessionIntent) -> datetime:
        # Canonical timestamp is carried by intent (authoritative upstream).
        return intent.ts_utc

    def _emit_intent_observed(self, intent: SessionIntent) -> None:
        # Optional; keep for now to validate end-to-end plumbing.
        self.sink.emit(
            ts_utc=intent.ts_utc,
            sim_time_s=float(self.env.now),
            event_type="session_intent",
            intent_id=intent.intent_id,
            intent_source=intent.intent_source,
            channel=intent.channel,
            audience_id=intent.audience_id,
            payload=intent.payload,
        )

    def _emit_user_created(self, *, ts_utc: datetime, user_id: str, intent: SessionIntent) -> None:
        self.sink.emit(
            ts_utc=ts_utc,
            sim_time_s=float(self.env.now),
            event_type="user_created",
            user_id=user_id,
            intent_id=intent.intent_id,
            intent_source=intent.intent_source,
            channel=intent.channel,
            audience_id=intent.audience_id,
            payload=intent.payload,
        )

    def _emit_session_start(
        self, *, ts_utc: datetime, user_id: str, session_id: str, intent: SessionIntent
    ) -> None:
        self.sink.emit(
            ts_utc=ts_utc,
            sim_time_s=float(self.env.now),
            event_type="session_start",
            user_id=user_id,
            session_id=session_id,
            intent_id=intent.intent_id,
            intent_source=intent.intent_source,
            channel=intent.channel,
            audience_id=intent.audience_id,
            payload=intent.payload,
        )

    def _resolve_user(self, *, ts_utc: datetime) -> tuple[str, bool]:
        """
        Uses UsersStateService public API:
          (user_state, is_new) = users.get_or_create_user_for_intent(...)
        """
        u, is_new = self.users.get_or_create_user_for_intent(now_utc=ts_utc, rng=self.rng)
        return u.user_id, is_new

    def _new_session_id(self) -> str:
        return self.ids.next_id("session")

    def _run_one(self, intent: SessionIntent) -> None:
        if not self.cfg.enabled:
            return

        ts_utc = self._now_ts_utc(intent)

        self._emit_intent_observed(intent)

        user_id, created = self._resolve_user(ts_utc=ts_utc)

        if created:
            self._emit_user_created(ts_utc=ts_utc, user_id=user_id, intent=intent)

        session_id = self._new_session_id()
        self._emit_session_start(
            ts_utc=ts_utc, user_id=user_id, session_id=session_id, intent=intent
        )

        # Spawn downstream session process
        self.session_runner.start_session(user_id=user_id, session_id=session_id, intent=intent)

    def _run_loop(self):
        while True:
            intent: SessionIntent = yield self.intents.get()
            self._run_one(intent)
