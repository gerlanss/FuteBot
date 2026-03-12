import json
import re
from typing import Any

import requests

from config import (
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    USE_GEMINI_MARKET_LOOKUP,
)

_TIMEOUT = 25
_BOOKMAKERS = ["Bet365", "Betano", "1xBet", "Pinnacle", "Sportingbet", "Betfair"]


class GeminiMarketLookup:
    def __init__(self):
        self.ativo = bool(GEMINI_API_KEY) and USE_GEMINI_MARKET_LOOKUP

    def lookup_market(self, oportunidade: dict) -> dict:
        debug = self.debug_lookup_market(oportunidade)
        if not debug.get("enabled"):
            return {"enabled": False}
        if debug.get("error"):
            return {
                "enabled": True,
                "error": debug["error"],
                "market_found": False,
                "bookmakers": [],
                "market_summary": "",
                "weather_summary": "",
                "field_conditions": "",
                "rotation_risk": "desconhecido",
                "motivation_context": "",
                "news_summary": "",
                "risk_flags": [],
                "context_summary": f"Erro no Gemini lookup: {debug['error']}",
                "sources": [],
            }
        parsed = debug.get("merged", {})
        parsed["enabled"] = True
        return parsed

    def debug_lookup_market(self, oportunidade: dict) -> dict:
        """Executa chamadas curtas por tema e retorna trilha detalhada."""
        if not self.ativo:
            return {"enabled": False, "error": "Gemini desativado"}

        stages = []
        merged = {
            "market_found": False,
            "bookmakers": [],
            "market_summary": "",
            "weather_summary": "",
            "field_conditions": "",
            "rotation_risk": "desconhecido",
            "motivation_context": "",
            "news_summary": "",
            "risk_flags": [],
            "context_summary": "",
            "sources": [],
        }

        for stage_name in ("market", "weather", "team_news"):
            prompt = self._build_prompt(oportunidade, stage_name)
            try:
                payload = self._call_gemini(prompt, stage_name)
                stage_debug = self._parse_stage_response(payload, stage_name)
                stage_debug["prompt"] = prompt
            except Exception as exc:
                stage_debug = {
                    "stage": stage_name,
                    "prompt": prompt,
                    "error": str(exc),
                    "raw_payload": {},
                    "raw_text": "",
                    "queries": [],
                    "sources": [],
                    "parsed_from": "error",
                    "parsed": {},
                }
            stages.append(stage_debug)
            self._merge_stage(merged, stage_name, stage_debug.get("parsed", {}))
            for source in stage_debug.get("sources", []):
                if source not in merged["sources"]:
                    merged["sources"].append(source)

        merged["context_summary"] = self._build_context_summary(merged)
        return {
            "enabled": True,
            "stages": stages,
            "merged": merged,
            "error": self._collect_error(stages),
        }

    def _build_prompt(self, oportunidade: dict, stage: str) -> str:
        base = (
            f"Jogo: {oportunidade.get('home_name', '?')} vs {oportunidade.get('away_name', '?')}. "
            f"Data: {oportunidade.get('date', '')}. "
            f"Liga ID: {oportunidade.get('league_id', '')}. "
            f"Mercado interno: {oportunidade.get('mercado', '?')}. "
            f"Descricao: {oportunidade.get('descricao', oportunidade.get('mercado', '?'))}."
        )
        if stage == "market":
            return (
                "Pesquise rapidamente na web se este mercado parece estar aberto em casas conhecidas. "
                "Nao invente odds. Responda SOMENTE em JSON puro e curto no formato "
                "{\"market_found\": true/false, \"bookmakers\": [\"casa1\",\"casa2\"], "
                "\"market_summary\": \"frase curta\", \"confidence\": 0.0}. "
                "Use no maximo 2 casas e uma frase de ate 12 palavras. "
                + base
            )
        if stage == "weather":
            return (
                "Pesquise na web apenas clima previsto e condicao do campo/gramado para este jogo. "
                "Responda SOMENTE em JSON puro e curto no formato "
                "{\"weather_summary\": \"texto curto\", \"field_conditions\": \"texto curto\", "
                "\"risk_flags\": [\"clima\"], \"confidence\": 0.0}. "
                "Cada campo deve ter no maximo 8 palavras. "
                + base
            )
        return (
            "Pesquise na web apenas noticias recentes, risco de rotacao/time misto e importancia competitiva do jogo. "
            "Responda SOMENTE em JSON puro e curto no formato "
            "{\"rotation_risk\": \"baixo|medio|alto|desconhecido\", "
            "\"motivation_context\": \"texto curto\", \"news_summary\": \"texto curto\", "
            "\"risk_flags\": [\"rotacao\"], \"confidence\": 0.0}. "
            "Use frases de no maximo 10 palavras e no maximo 2 risk_flags. "
            + base
        )

    def _call_gemini(self, prompt: str, stage: str) -> dict[str, Any]:
        url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent"
        headers = {
            "x-goog-api-key": GEMINI_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 220 if stage != "team_news" else 320,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        last_exc = None
        for max_tokens in (payload["generationConfig"]["maxOutputTokens"], 160):
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                body = ""
                if hasattr(exc, "response") and exc.response is not None:
                    body = exc.response.text[:500]
                if body:
                    last_exc = RuntimeError(f"{exc} | body={body}")
                else:
                    last_exc = exc
        raise last_exc

    def _parse_stage_response(self, payload: dict, stage: str) -> dict:
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ValueError("resposta vazia do Gemini")

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()
        grounding = candidate.get("groundingMetadata", {}) or {}
        queries = (grounding.get("webSearchQueries") or [])[:10]

        parsed_from = "json"
        if text:
            try:
                data = self._extract_json(text)
            except Exception:
                parsed_from = "grounding_fallback"
                data = self._fallback_from_grounding(text, grounding, stage)
        else:
            parsed_from = "grounding_fallback"
            data = self._fallback_from_grounding(text, grounding, stage)

        sources = self._extract_sources(grounding)
        parsed = self._normalize_stage(data, text, grounding, stage)
        return {
            "stage": stage,
            "raw_payload": payload,
            "raw_text": text,
            "queries": queries,
            "sources": sources,
            "parsed_from": parsed_from,
            "parsed": parsed,
        }

    def _extract_json(self, text: str) -> dict:
        clean = text.strip()
        if clean.startswith("```"):
            lines = [line for line in clean.splitlines() if not line.strip().startswith("```")]
            clean = "\n".join(lines).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{.*?\}", clean, re.DOTALL)
            if not match:
                raise
            return json.loads(match.group())

    def _extract_sources(self, grounding: dict) -> list[dict]:
        sources = []
        for chunk in grounding.get("groundingChunks", []) or []:
            web = chunk.get("web", {}) or {}
            uri = web.get("uri")
            if uri:
                sources.append({
                    "title": web.get("title", uri),
                    "url": uri,
                })
        return sources[:8]

    def _fallback_from_grounding(self, text: str, grounding: dict, stage: str) -> dict:
        queries = grounding.get("webSearchQueries", []) or []
        combined = " ".join([text or "", *queries]).lower()

        if stage == "market":
            houses = [book for book in _BOOKMAKERS if book.lower() in combined]
            return {
                "market_found": bool(houses or queries),
                "bookmakers": houses[:2],
                "market_summary": (
                    "Mercado encontrado em buscas externas."
                    if houses or queries else
                    "Buscas nao confirmaram mercado."
                ),
                "confidence": 0.6 if houses else 0.4,
            }

        if stage == "weather":
            flags = []
            if "weather" in combined or "tempo" in combined:
                flags.append("clima")
            if "rain" in combined or "chuva" in combined:
                flags.append("chuva")
            return {
                "weather_summary": self._infer_weather_summary(combined),
                "field_conditions": self._infer_field_conditions(combined),
                "risk_flags": flags[:2],
                "confidence": 0.4,
            }

        rotation = "desconhecido"
        flags = []
        if "reserve" in combined or "rotation" in combined or "rotacao" in combined:
            rotation = "medio"
            flags.append("rotacao")
        if "injury" in combined or "les" in combined or "suspens" in combined:
            flags.append("lesoes")
        if "relegation" in combined or "title" in combined or "import" in combined:
            flags.append("motivacao")
        return {
            "rotation_risk": rotation,
            "motivation_context": self._infer_motivation_context(combined),
            "news_summary": self._infer_news_summary(combined),
            "risk_flags": flags[:2],
            "confidence": 0.4,
        }

    def _normalize_stage(self, data: dict, text: str, grounding: dict, stage: str) -> dict:
        if stage == "market":
            bookmakers = self._normalize_list(data.get("bookmakers"))
            if not bookmakers:
                combined = " ".join([text or "", *(grounding.get("webSearchQueries") or [])]).lower()
                bookmakers = [book for book in _BOOKMAKERS if book.lower() in combined]
            return {
                "market_found": bool(data.get("market_found", False) or bookmakers),
                "bookmakers": bookmakers[:2],
                "market_summary": self._trim_text(data.get("market_summary", ""), 90),
                "confidence": self._normalize_confidence(data.get("confidence")),
            }

        if stage == "weather":
            return {
                "weather_summary": self._trim_text(data.get("weather_summary", ""), 80),
                "field_conditions": self._trim_text(data.get("field_conditions", ""), 80),
                "risk_flags": self._normalize_list(data.get("risk_flags"))[:2],
                "confidence": self._normalize_confidence(data.get("confidence")),
            }

        return {
            "rotation_risk": self._normalize_rotation(data.get("rotation_risk")),
            "motivation_context": self._trim_text(data.get("motivation_context", ""), 90),
            "news_summary": self._trim_text(data.get("news_summary", ""), 90),
            "risk_flags": self._normalize_list(data.get("risk_flags"))[:2],
            "confidence": self._normalize_confidence(data.get("confidence")),
        }

    def _normalize(self, data: dict, text: str, grounding: dict) -> dict:
        """Compatibilidade com a API antiga usada nos testes e na auditoria."""
        merged = {
            "market_found": False,
            "bookmakers": [],
            "market_summary": "",
            "weather_summary": "",
            "field_conditions": "",
            "rotation_risk": "desconhecido",
            "motivation_context": "",
            "news_summary": "",
            "risk_flags": [],
            "context_summary": "",
            "sources": self._extract_sources(grounding),
        }
        self._merge_stage(merged, "market", self._normalize_stage(data, text, grounding, "market"))
        self._merge_stage(merged, "weather", self._normalize_stage(data, text, grounding, "weather"))
        self._merge_stage(merged, "team_news", self._normalize_stage(data, text, grounding, "team_news"))
        merged["context_summary"] = self._trim_text(
            data.get("context_summary") or self._build_context_summary(merged),
            220,
        )
        return merged

    def _merge_stage(self, merged: dict, stage: str, parsed: dict):
        if stage == "market":
            merged["market_found"] = bool(parsed.get("market_found", merged["market_found"]))
            merged["bookmakers"] = parsed.get("bookmakers") or merged["bookmakers"]
            merged["market_summary"] = parsed.get("market_summary") or merged["market_summary"]
            return

        if stage == "weather":
            merged["weather_summary"] = parsed.get("weather_summary") or merged["weather_summary"]
            merged["field_conditions"] = parsed.get("field_conditions") or merged["field_conditions"]
            merged["risk_flags"] = self._merge_flags(merged["risk_flags"], parsed.get("risk_flags"))
            return

        merged["rotation_risk"] = self._pick_rotation(merged["rotation_risk"], parsed.get("rotation_risk"))
        merged["motivation_context"] = parsed.get("motivation_context") or merged["motivation_context"]
        merged["news_summary"] = parsed.get("news_summary") or merged["news_summary"]
        merged["risk_flags"] = self._merge_flags(merged["risk_flags"], parsed.get("risk_flags"))

    def _build_context_summary(self, merged: dict) -> str:
        parts = [
            merged.get("weather_summary"),
            merged.get("field_conditions"),
            merged.get("motivation_context"),
            merged.get("news_summary"),
            merged.get("market_summary"),
        ]
        summary = " ".join(part for part in parts if part).strip()
        if summary:
            return summary
        return "Gemini nao retornou fatos externos estruturados suficientes."

    @staticmethod
    def _infer_weather_summary(combined: str) -> str:
        if "chuva" in combined or "rain" in combined:
            return "Buscas indicam chance de chuva."
        if "sol" in combined or "sunny" in combined:
            return "Buscas sugerem tempo firme."
        if "weather" in combined or "tempo" in combined:
            return "Buscas localizaram clima, sem detalhe confiavel."
        return ""

    @staticmethod
    def _infer_field_conditions(combined: str) -> str:
        if "gramado" in combined or "pitch" in combined or "field" in combined:
            return "Buscas mencionam gramado/campo."
        return ""

    @staticmethod
    def _infer_motivation_context(combined: str) -> str:
        if "title" in combined or "titulo" in combined:
            return "Buscas sugerem jogo importante por titulo."
        if "relegation" in combined or "rebaix" in combined:
            return "Buscas sugerem pressao por rebaixamento."
        if "import" in combined:
            return "Buscas apontam relevancia competitiva."
        return ""

    @staticmethod
    def _infer_news_summary(combined: str) -> str:
        if "injury" in combined or "les" in combined or "suspens" in combined:
            return "Buscas trouxeram sinais de desfalques recentes."
        if "rotation" in combined or "reserve" in combined or "rotacao" in combined:
            return "Buscas sugerem risco de rotacao."
        return ""

    @staticmethod
    def _collect_error(stages: list[dict]) -> str:
        errors = [stage.get("error") for stage in stages if stage.get("error")]
        return " | ".join(errors)

    @staticmethod
    def _merge_flags(current: list[str], incoming: Any) -> list[str]:
        merged = list(current or [])
        for item in incoming or []:
            label = str(item).strip()
            if label and label not in merged:
                merged.append(label)
        return merged

    @staticmethod
    def _pick_rotation(current: str, incoming: Any) -> str:
        order = {"desconhecido": 0, "baixo": 1, "medio": 2, "alto": 3}
        inc = GeminiMarketLookup._normalize_rotation(incoming)
        cur = GeminiMarketLookup._normalize_rotation(current)
        return inc if order.get(inc, 0) > order.get(cur, 0) else cur

    @staticmethod
    def _normalize_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if value:
            return [str(value).strip()]
        return []

    @staticmethod
    def _normalize_rotation(value: Any) -> str:
        val = str(value or "").strip().lower()
        if val in {"baixo", "medio", "alto", "desconhecido"}:
            return val
        return "desconhecido"

    @staticmethod
    def _normalize_confidence(value: Any) -> float:
        try:
            conf = float(value)
        except (TypeError, ValueError):
            conf = 0.4
        return max(0.0, min(1.0, conf))

    @staticmethod
    def _trim_text(value: Any, limit: int) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."
