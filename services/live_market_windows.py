from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiveMarketWindow:
    start_minute: int
    end_minute: int


def _resolve_window(mercado: str) -> LiveMarketWindow:
    market = (mercado or "").lower()
    ht_market = market.endswith("_ht") or market.startswith("ht_")
    second_half_market = market.endswith("_2t")
    is_corners = "corners" in market
    is_resultado = market.startswith("h2h_") or market.startswith("ht_")
    is_over = "over" in market
    is_under = "under" in market
    is_draw = market.endswith("draw")

    if ht_market:
        if is_corners and is_over:
            return LiveMarketWindow(20, 37)
        if is_corners and is_under:
            return LiveMarketWindow(24, 41)
        if is_resultado and is_draw:
            return LiveMarketWindow(26, 40)
        if is_resultado:
            return LiveMarketWindow(22, 38)
        if is_over:
            return LiveMarketWindow(18, 38)
        if is_under:
            return LiveMarketWindow(24, 41)
        return LiveMarketWindow(20, 39)

    if second_half_market:
        if is_corners and is_over:
            return LiveMarketWindow(52, 78)
        if is_corners and is_under:
            return LiveMarketWindow(60, 84)
        if is_resultado and is_draw:
            return LiveMarketWindow(58, 80)
        if is_resultado:
            return LiveMarketWindow(52, 78)
        if is_over:
            return LiveMarketWindow(52, 78)
        if is_under:
            return LiveMarketWindow(60, 84)
        return LiveMarketWindow(54, 80)

    if is_corners and is_over:
        return LiveMarketWindow(50, 78)
    if is_corners and is_under:
        return LiveMarketWindow(58, 82)
    if is_resultado and is_draw:
        return LiveMarketWindow(58, 80)
    if is_resultado:
        return LiveMarketWindow(52, 78)
    if is_over:
        return LiveMarketWindow(50, 78)
    if is_under:
        return LiveMarketWindow(58, 84)
    return LiveMarketWindow(50, 80)


def janela_operacional_live(mercado: str) -> tuple[int, int]:
    window = _resolve_window(mercado)
    return window.start_minute, window.end_minute


def status_janela_operacional_live(mercado: str, elapsed: int | str | None) -> tuple[str, str]:
    try:
        minute = int(elapsed or 0)
    except Exception:
        return "ok", ""
    start_minute, end_minute = janela_operacional_live(mercado)
    if minute < start_minute:
        return "cedo", (
            f"A leitura ainda apareceu cedo demais para este mercado "
            f"(janela útil a partir de {start_minute}')."
        )
    if minute >= end_minute:
        return "tarde", (
            f"A leitura apareceu tarde demais para este mercado "
            f"(janela útil até {end_minute}')."
        )
    return "ok", ""


def dentro_janela_operacional_live(mercado: str, elapsed: int | str | None) -> bool:
    status, _ = status_janela_operacional_live(mercado, elapsed)
    return status == "ok"
