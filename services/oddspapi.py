"""
Client OddsPapi v4 focado no fluxo operacional do FuteBot.

Usa exclusivamente a 1xBet e só entra no funil final:
  - pré-live: shortlist aprovada pelo scanner/T-30
  - live: fixtures já monitorados
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import unicodedata
from typing import Any

import requests

from config import (
    ODDSPAPI_BASE,
    ODDSPAPI_BOOKMAKER_LABEL,
    ODDSPAPI_BOOKMAKER_SLUG,
    ODDSPAPI_KEY,
)

_TIMEOUT = 20
_SOCCER_SPORT_ID = 10


class OddsPapiError(RuntimeError):
    """Erro genérico da OddsPapi."""


class OddsPapiRestrictedAccess(OddsPapiError):
    """A conta não possui acesso live/liberado para o bookmaker consultado."""


class OddsPapiMissingFixture(OddsPapiError):
    """Não foi possível resolver o fixture correspondente na OddsPapi."""


@dataclass(frozen=True)
class OddsSelection:
    market_id: int
    outcome_id: int
    label: str


_TIP_TO_SELECTION: dict[str, OddsSelection] = {
    "h2h_home": OddsSelection(101, 101, "Full Time Result: 1"),
    "h2h_draw": OddsSelection(101, 102, "Full Time Result: X"),
    "h2h_away": OddsSelection(101, 103, "Full Time Result: 2"),
    "over15": OddsSelection(108, 108, "Over 1.5"),
    "under15": OddsSelection(108, 109, "Under 1.5"),
    "over25": OddsSelection(1010, 1010, "Over 2.5"),
    "under25": OddsSelection(1010, 1011, "Under 2.5"),
    "over35": OddsSelection(1012, 1012, "Over 3.5"),
    "under35": OddsSelection(1012, 1013, "Under 3.5"),
    "ht_home": OddsSelection(10208, 10208, "1T Resultado: 1"),
    "ht_draw": OddsSelection(10208, 10209, "1T Resultado: X"),
    "ht_away": OddsSelection(10208, 10210, "1T Resultado: 2"),
    "over05_ht": OddsSelection(10256, 10256, "1T Over 0.5"),
    "under05_ht": OddsSelection(10256, 10257, "1T Under 0.5"),
    "over15_ht": OddsSelection(10258, 10258, "1T Over 1.5"),
    "under15_ht": OddsSelection(10258, 10259, "1T Under 1.5"),
    "over05_2t": OddsSelection(10270, 10270, "2T Over 0.5"),
    "under05_2t": OddsSelection(10270, 10271, "2T Under 0.5"),
    "over15_2t": OddsSelection(10272, 10272, "2T Over 1.5"),
    "under15_2t": OddsSelection(10272, 10273, "2T Under 1.5"),
}


def _normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = "".join(ch for ch in text if not unicodedata.combining(ch))
    cleaned = []
    for ch in ascii_text.lower():
        cleaned.append(ch if ch.isalnum() else " ")
    return " ".join("".join(cleaned).split())


def _words_for_match(value: str) -> list[str]:
    words = [w for w in _normalize_name(value).split() if len(w) >= 3]
    return words or _normalize_name(value).split()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


class OddsPapiClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        bookmaker_slug: str | None = None,
    ):
        self.api_key = (api_key or ODDSPAPI_KEY or "").strip()
        self.base_url = (base_url or ODDSPAPI_BASE).rstrip("/")
        self.bookmaker_slug = (bookmaker_slug or ODDSPAPI_BOOKMAKER_SLUG).strip() or "1xbet"

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        if not self.api_key:
            raise OddsPapiError("ODDSPAPI_KEY ausente.")
        query = dict(params or {})
        query["apiKey"] = self.api_key
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, params=query, timeout=_TIMEOUT)
        if response.status_code == 403 and "RESTRICTED_ACCESS" in response.text:
            raise OddsPapiRestrictedAccess(response.text)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise OddsPapiError(f"OddsPapi {response.status_code}: {response.text[:300]}") from exc
        return response.json()

    def fixtures(self, **params: Any) -> list[dict]:
        data = self._request("/fixtures", params)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        return []

    def odds(self, fixture_id: str, *, live: bool | None = None) -> dict:
        params: dict[str, Any] = {
            "fixtureId": fixture_id,
            "bookmakers": self.bookmaker_slug,
        }
        if live is True:
            params["isLive"] = "true"
        return self._request("/odds", params)

    def historical_odds(self, fixture_id: str) -> dict:
        return self._request(
            "/historical-odds",
            {"fixtureId": fixture_id, "bookmakers": self.bookmaker_slug},
        )

    def markets(self, **params: Any) -> list[dict]:
        data = self._request("/markets", params)
        return data if isinstance(data, list) else []

    def resolve_fixture(self, tip: dict, *, phase_operacional: str = "prelive_final") -> dict:
        kickoff = _parse_iso(tip.get("date"))
        if kickoff is None:
            raise OddsPapiMissingFixture("Tip sem kickoff válido para casar fixture.")

        from_dt = kickoff.astimezone(timezone.utc) - timedelta(hours=4)
        to_dt = kickoff.astimezone(timezone.utc) + timedelta(hours=4)
        params: dict[str, Any] = {
            "sportId": _SOCCER_SPORT_ID,
            "from": from_dt.isoformat().replace("+00:00", "Z"),
            "to": to_dt.isoformat().replace("+00:00", "Z"),
            "hasOdds": "true",
        }
        if phase_operacional.startswith("live"):
            params["statusId"] = 1

        candidatos = self.fixtures(**params)
        home_words = _words_for_match(tip.get("home_name"))
        away_words = _words_for_match(tip.get("away_name"))

        melhor: tuple[int, dict] | None = None
        for fixture in candidatos:
            nome1 = _normalize_name(fixture.get("participant1Name"))
            nome2 = _normalize_name(fixture.get("participant2Name"))
            home_hits = sum(1 for word in home_words if word in nome1)
            away_hits = sum(1 for word in away_words if word in nome2)
            score = home_hits + away_hits
            if score == 0:
                reverse_home_hits = sum(1 for word in home_words if word in nome2)
                reverse_away_hits = sum(1 for word in away_words if word in nome1)
                score = reverse_home_hits + reverse_away_hits
            if score <= 0:
                continue
            if melhor is None or score > melhor[0]:
                melhor = (score, fixture)

        if melhor is None:
            raise OddsPapiMissingFixture(
                f"Fixture não encontrado para {tip.get('home_name')} x {tip.get('away_name')}"
            )
        return melhor[1]

    def extract_price(self, odds_payload: dict, mercado: str) -> dict | None:
        selection = _TIP_TO_SELECTION.get(mercado)
        if not selection:
            return None

        bookmaker_data = ((odds_payload or {}).get("bookmakerOdds") or {}).get(self.bookmaker_slug) or {}
        markets = bookmaker_data.get("markets") or {}
        market_payload = markets.get(str(selection.market_id)) or {}
        outcomes = market_payload.get("outcomes") or {}
        outcome_payload = outcomes.get(str(selection.outcome_id)) or {}
        players = outcome_payload.get("players") or {}
        entry = players.get("0")

        if isinstance(entry, list):
            entry = entry[0] if entry else None
        elif isinstance(entry, dict):
            pass
        elif isinstance(players, list):
            entry = players[0] if players else None
        else:
            for candidate in players.values():
                if isinstance(candidate, list) and candidate:
                    entry = candidate[0]
                    break
                if isinstance(candidate, dict):
                    entry = candidate
                    break

        if not isinstance(entry, dict):
            return None

        price = entry.get("price")
        if price is None:
            return None

        changed_at = entry.get("changedAt") or entry.get("createdAt")
        active = entry.get("active")
        return {
            "odd": float(price),
            "changed_at": changed_at,
            "active": bool(active) if active is not None else None,
            "selection_label": selection.label,
            "fixture_path": bookmaker_data.get("fixturePath"),
            "bookmaker_fixture_id": bookmaker_data.get("bookmakerFixtureId"),
        }


def enriquecer_tips_com_odds_oddspapi(
    tips: list[dict],
    *,
    phase_operacional: str = "prelive_final",
    client: OddsPapiClient | None = None,
) -> list[dict]:
    """
    Enriquece a shortlist final com odds OddsPapi/1xBet.

    Não consulta universo amplo; só resolve e consulta os fixtures passados.
    """
    if not tips:
        return tips

    api = client or OddsPapiClient()
    fixture_cache: dict[tuple[int | None, str, str, str], dict | None] = {}
    odds_cache: dict[str, dict | Exception] = {}

    for tip in tips:
        tip.setdefault("odds_provider", "oddspapi_v4")
        tip.setdefault("bookmaker", ODDSPAPI_BOOKMAKER_LABEL)
        tip.setdefault("odd_capture_type", "live" if phase_operacional.startswith("live") else "pregame")
        tip.setdefault("odd_status", "pending")

        key = (
            tip.get("fixture_id"),
            str(tip.get("date") or ""),
            str(tip.get("home_name") or ""),
            str(tip.get("away_name") or ""),
        )
        resolved = fixture_cache.get(key)
        if resolved is None and key not in fixture_cache:
            try:
                resolved = api.resolve_fixture(tip, phase_operacional=phase_operacional)
            except OddsPapiMissingFixture:
                resolved = None
            fixture_cache[key] = resolved

        if not resolved:
            tip["odd_status"] = "fixture_nao_encontrado"
            tip["odd_block_reason"] = "fixture_sem_odd"
            tip["acesso_live_disponivel"] = False if phase_operacional.startswith("live") else None
            continue

        oddspapi_fixture_id = resolved.get("fixtureId")
        if not oddspapi_fixture_id:
            tip["odd_status"] = "fixture_nao_encontrado"
            tip["odd_block_reason"] = "fixture_sem_odd"
            continue

        tip["oddspapi_fixture_id"] = oddspapi_fixture_id
        tip["oddspapi_status_id"] = resolved.get("statusId")

        payload = odds_cache.get(oddspapi_fixture_id)
        if payload is None and oddspapi_fixture_id not in odds_cache:
            try:
                payload = api.odds(oddspapi_fixture_id, live=phase_operacional.startswith("live"))
            except OddsPapiRestrictedAccess as exc:
                payload = exc
            except OddsPapiError as exc:
                payload = exc
            odds_cache[oddspapi_fixture_id] = payload

        if isinstance(payload, OddsPapiRestrictedAccess):
            tip["odd_status"] = "acesso_live_restrito"
            tip["odd_block_reason"] = "acesso_live_restrito"
            tip["acesso_live_disponivel"] = False
            continue

        if isinstance(payload, Exception):
            tip["odd_status"] = "erro_consulta"
            tip["odd_block_reason"] = "fixture_sem_odd"
            tip["acesso_live_disponivel"] = False if phase_operacional.startswith("live") else None
            continue

        tip["acesso_live_disponivel"] = True if phase_operacional.startswith("live") else None
        detalhe = api.extract_price(payload, tip.get("mercado", ""))
        if not detalhe:
            tip["odd_status"] = "sem_odd_valida"
            tip["odd_block_reason"] = "fixture_sem_odd"
            continue

        odd = round(float(detalhe["odd"]), 2)
        tip["odd_status"] = "ok"
        tip["odd_usada"] = odd
        tip["odd_fonte"] = ODDSPAPI_BOOKMAKER_LABEL
        tip["bookmaker"] = ODDSPAPI_BOOKMAKER_LABEL
        tip["odd_captured_at"] = detalhe.get("changed_at")
        tip["odd_selection_label"] = detalhe.get("selection_label")
        tip["odd_fixture_path"] = detalhe.get("fixture_path")
        tip["odd_bookmaker_fixture_id"] = detalhe.get("bookmaker_fixture_id")
        prob = float(tip.get("prob_modelo") or 0)
        tip["ev_percent"] = round((prob * odd - 1) * 100, 1)

    return tips
