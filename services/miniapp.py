"""
Mini App HTTP para o Telegram.

Serve uma interface leve de status e preferencias do bot.
O acesso no Telegram depende de uma URL HTTPS publica configurada em MINI_APP_URL.
"""

from __future__ import annotations

from pathlib import Path

from aiohttp import web

from config import LEAGUES, MINI_APP_BIND_HOST, MINI_APP_BIND_PORT
from services.user_prefs import load_preferences, save_preferences


BASE_DIR = Path(__file__).resolve().parent.parent
MINIAPP_DIR = BASE_DIR / "miniapp"


class MiniAppServer:
    def __init__(self, db):
        self.db = db
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self.app = web.Application()
        self.app.router.add_get("/miniapp", self._index)
        self.app.router.add_get("/miniapp/", self._index)
        self.app.router.add_get("/miniapp/assets/{name}", self._asset)
        self.app.router.add_get("/miniapp/api/state", self._state)
        self.app.router.add_post("/miniapp/api/preferences", self._save_preferences)

    async def start(self):
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        self._site = web.TCPSite(
            self._runner,
            host=MINI_APP_BIND_HOST,
            port=MINI_APP_BIND_PORT,
        )
        await self._site.start()
        print(f"[MiniApp] HTTP ativo em http://{MINI_APP_BIND_HOST}:{MINI_APP_BIND_PORT}/miniapp")

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None

    async def _index(self, request: web.Request) -> web.Response:
        return web.FileResponse(MINIAPP_DIR / "index.html")

    async def _asset(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        return web.FileResponse(MINIAPP_DIR / name)

    async def _state(self, request: web.Request) -> web.Response:
        chat_id = request.query.get("chat_id", "").strip()
        prefs = load_preferences().get(chat_id, {}) if chat_id else {}

        resumo = self.db.resumo()
        treino = self.db.ultimo_treino()
        metricas = self.db.metricas_modelo()
        ultimas = self._ultimas_predictions(limit=60)
        markets = sorted({item["mercado"] for item in ultimas if item.get("mercado")})
        bankroll = self._build_bankroll(prefs, metricas)

        payload = {
            "chat_id": chat_id,
            "summary": resumo,
            "metrics": metricas,
            "training": treino or {},
            "preferences": {
                "alerts_enabled": prefs.get("alerts_enabled", True),
                "min_ev": prefs.get("min_ev", 3.0),
                "favorite_leagues": prefs.get("favorite_leagues", []),
                "bankroll_initial": prefs.get("bankroll_initial", 1000.0),
                "stake_unit": prefs.get("stake_unit", 1.0),
            },
            "leagues": [
                {"id": liga["id"], "key": key, "name": liga["nome"]}
                for key, liga in LEAGUES.items()
            ],
            "markets": markets,
            "bankroll": bankroll,
            "latest_predictions": ultimas,
        }
        return web.json_response(payload)

    async def _save_preferences(self, request: web.Request) -> web.Response:
        data = await request.json()
        chat_id = str(data.get("chat_id", "")).strip()
        if not chat_id:
            return web.json_response({"ok": False, "error": "chat_id obrigatorio"}, status=400)

        prefs = load_preferences()
        prefs[chat_id] = {
            "alerts_enabled": bool(data.get("alerts_enabled", True)),
            "min_ev": float(data.get("min_ev", 3.0)),
            "favorite_leagues": [int(x) for x in data.get("favorite_leagues", [])],
            "bankroll_initial": float(data.get("bankroll_initial", 1000.0)),
            "stake_unit": float(data.get("stake_unit", 1.0)),
        }
        save_preferences(prefs)
        return web.json_response({"ok": True})

    def _ultimas_predictions(self, limit: int = 12) -> list[dict]:
        conn = self.db._conn()
        rows = conn.execute(
            """
            SELECT date, league_id, home_name, away_name, mercado,
                   odd_usada, ev_percent, bookmaker, acertou, lucro
            FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()

        itens = []
        for row in rows:
            itens.append({
                "date": row["date"],
                "league_id": row["league_id"],
                "league_name": self._league_name(row["league_id"]),
                "home_name": row["home_name"],
                "away_name": row["away_name"],
                "mercado": row["mercado"],
                "odd": row["odd_usada"],
                "ev_percent": row["ev_percent"],
                "bookmaker": row["bookmaker"],
                "lucro": row["lucro"],
                "status": "win" if row["acertou"] == 1 else "loss" if row["acertou"] == 0 else "open",
            })
        return itens

    def _league_name(self, league_id: int | None) -> str:
        if league_id is None:
            return "Liga"
        for liga in LEAGUES.values():
            if liga["id"] == league_id:
                return liga["nome"]
        return f"Liga {league_id}"

    def _build_bankroll(self, prefs: dict, metricas: dict) -> dict:
        inicial = float(prefs.get("bankroll_initial", 1000.0))
        stake_unit = float(prefs.get("stake_unit", 1.0))
        lucro_unidades = float(metricas.get("lucro_total", 0) or 0)
        lucro_monetario = lucro_unidades * stake_unit
        atual = inicial + lucro_monetario
        roi = float(metricas.get("roi", 0) or 0)
        return {
            "initial": round(inicial, 2),
            "current": round(atual, 2),
            "stake_unit": round(stake_unit, 2),
            "profit_units": round(lucro_unidades, 2),
            "profit_value": round(lucro_monetario, 2),
            "roi_percent": round(roi, 1),
        }
