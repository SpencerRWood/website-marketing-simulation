from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import simpy

from sim.core.ids import IdsService
from sim.core.rng import RNG
from sim.features.intent_resolver.types import EventSink, IntentResolverConfig, SessionRunner
from sim.features.session_intent.types import SessionIntent
from sim.features.users_state.service import UsersStateService


@dataclass(slots=True)
class IntentResolverService:
    """Consumes SessionIntent messages and resolves them into (user_id, session_id).

    Responsibilities:
      - Delegate user creation/selection to UsersStateService (hot storage)
      - Generate session IDs
      - Emit events (EventSink)
      - Spawn downstream session runner process
    """

    env: simpy.Environment
    cfg: IntentResolverConfig

    ids: IdsService
    rng: RNG

    # Upstream: expects SessionIntentService with .get()
    intents: Any

    # Hot state (UsersStateService-compatible)
    users: UsersStateService

    # Downstream
    session_runner: SessionRunner

    # Events
    sink: EventSink

    def start(self) -> simpy.events.Process:
        return self.env.process(self._run_loop())

    def _emit(
        self,
        event_type: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        intent: SessionIntent | None = None,
    ) -> None:
        intent_source = getattr(intent, "intent_source", None) if intent is not None else None
        channel = getattr(intent, "channel", None) if intent is not None else None
        channel_out = channel or "direct"

        payload: dict[str, Any] | None = None
        if intent is not None:
            payload = dict(getattr(intent, "payload", None) or {})
            # Keep schema stable: store optional fields in payload_json
            if getattr(intent, "intent_id", None) is not None:
                payload["intent_id"] = intent.intent_id
            if getattr(intent, "audience_id", None) is not None:
                payload["audience_id"] = intent.audience_id
            if not payload:
                payload = None

        self.sink.emit(
            event_type,
            user_id=user_id,
            session_id=session_id,
            intent_source=intent_source,
            channel=channel_out
            if session_id is not None
            else channel,  # session-scoped needs channel
            payload=payload,
        )

    def _resolve_user(self, *, ts_utc) -> tuple[str, bool]:
        u, is_new = self.users.get_or_create_user_for_intent(now_utc=ts_utc, rng=self.rng)
        return u.user_id, is_new

    def _new_session_id(self) -> str:
        return self.ids.next_id("session")

    def _run_one(self, intent: SessionIntent) -> None:
        if not self.cfg.enabled:
            return

        # Canonical timestamp is carried by intent; used for user-state recency.
        ts_utc = intent.ts_utc

        self._emit("session_intent", intent=intent)

        user_id, created = self._resolve_user(ts_utc=ts_utc)

        if created:
            self._emit("user_created", user_id=user_id, intent=intent)

        session_id = self._new_session_id()
        self._emit("session_start", user_id=user_id, session_id=session_id, intent=intent)

        self.session_runner.start_session(user_id=user_id, session_id=session_id, intent=intent)

    def _run_loop(self):
        while True:
            intent: SessionIntent = yield self.intents.get()
            self._run_one(intent)
