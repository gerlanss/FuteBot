"""
Módulo de banco de dados SQLite do FuteBot.

Gerencia todas as tabelas usadas pelo sistema:
  - fixtures: jogos (agenda + resultados)
  - fixture_stats: estatísticas de cada partida (chutes, posse, xG, etc.)
  - team_stats: estatísticas agregadas de um time em uma liga/season
  - predictions: previsões do modelo para jogos futuros
  - odds_cache: odds buscadas do The Odds API
  - model_scores: performance de cada estratégia/modelo por liga
  - train_log: log de treinamentos do XGBoost

Uso:
  from data.database import Database
  db = Database()
  db.salvar_fixture({...})
  jogos = db.fixtures_por_liga(71, 2024)
"""

import sqlite3
import json
import os
from datetime import datetime
from config import DB_PATH


class Database:
    """Interface SQLite centralizada. Abre conexão por chamada (thread-safe)."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # Garante que a pasta data/ existe
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._criar_tabelas()

    # ══════════════════════════════════════════════
    #  CONEXÃO E SCHEMA
    # ══════════════════════════════════════════════

    def _conn(self) -> sqlite3.Connection:
        """Abre conexão com Row factory para acesso por nome de coluna."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Performance para escrita concorrente
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _criar_tabelas(self):
        """Cria todas as tabelas se não existirem."""
        conn = self._conn()
        conn.executescript("""
            -- Jogos (fixtures): agenda, resultado, liga, temporada
            CREATE TABLE IF NOT EXISTS fixtures (
                fixture_id   INTEGER PRIMARY KEY,
                league_id    INTEGER NOT NULL,
                league_name  TEXT,
                season       INTEGER NOT NULL,
                round        TEXT,
                date         TEXT NOT NULL,           -- ISO 8601
                timestamp    INTEGER,
                venue        TEXT,
                status       TEXT DEFAULT 'NS',       -- NS, FT, LIVE, etc.
                home_id      INTEGER NOT NULL,
                home_name    TEXT NOT NULL,
                away_id      INTEGER NOT NULL,
                away_name    TEXT NOT NULL,
                goals_home   INTEGER,                 -- NULL se não jogou
                goals_away   INTEGER,
                score_ht_h   INTEGER,                 -- Placar do 1o tempo
                score_ht_a   INTEGER,
                referee      TEXT,
                raw_json     TEXT,                     -- JSON completo da API
                created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at   TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Índices para buscas rápidas
            CREATE INDEX IF NOT EXISTS idx_fixtures_league_season
                ON fixtures(league_id, season);
            CREATE INDEX IF NOT EXISTS idx_fixtures_date
                ON fixtures(date);
            CREATE INDEX IF NOT EXISTS idx_fixtures_status
                ON fixtures(status);
            CREATE INDEX IF NOT EXISTS idx_fixtures_teams
                ON fixtures(home_id, away_id);

            -- Estatísticas por partida (chutes, posse, xG, etc.)
            CREATE TABLE IF NOT EXISTS fixture_stats (
                fixture_id   INTEGER NOT NULL,
                team_id      INTEGER NOT NULL,
                team_name    TEXT,
                stats_json   TEXT NOT NULL,            -- JSON com todas as stats
                -- Campos mais usados extraídos para consulta rápida
                shots_total      INTEGER,
                shots_on_target  INTEGER,
                possession       REAL,                -- Percentual (ex: 55.0)
                passes_total     INTEGER,
                passes_accuracy  REAL,
                fouls            INTEGER,
                corners          INTEGER,
                yellow_cards     INTEGER,
                red_cards        INTEGER,
                expected_goals   REAL,                -- xG
                created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (fixture_id, team_id),
                FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
            );

            -- Eventos de partida (gols, cartões, substituições)
            CREATE TABLE IF NOT EXISTS fixture_events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id   INTEGER NOT NULL,
                team_id      INTEGER,
                player_name  TEXT,
                event_type   TEXT,                     -- Goal, Card, Subst
                event_detail TEXT,                     -- Normal Goal, Yellow Card, etc.
                minute       INTEGER,
                extra_minute INTEGER,
                raw_json     TEXT,
                FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
            );

            CREATE INDEX IF NOT EXISTS idx_events_fixture
                ON fixture_events(fixture_id);

            -- Estatísticas agregadas do time (por liga/season)
            CREATE TABLE IF NOT EXISTS team_stats (
                team_id      INTEGER NOT NULL,
                league_id    INTEGER NOT NULL,
                season       INTEGER NOT NULL,
                team_name    TEXT,
                form         TEXT,                     -- Ex: 'WDWLW'
                stats_json   TEXT NOT NULL,            -- JSON completo
                -- Campos extraídos para consulta rápida
                played       INTEGER,
                wins         INTEGER,
                draws        INTEGER,
                losses       INTEGER,
                goals_for    INTEGER,
                goals_against INTEGER,
                goals_for_avg    REAL,
                goals_against_avg REAL,
                clean_sheets INTEGER,
                failed_to_score INTEGER,
                created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at   TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, league_id, season)
            );

            -- Previsões do modelo (para tracking e aprendizado)
            CREATE TABLE IF NOT EXISTS predictions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id   INTEGER NOT NULL,
                date         TEXT NOT NULL,
                league_id    INTEGER,
                home_name    TEXT,
                away_name    TEXT,
                -- Probabilidades do modelo
                prob_home    REAL,
                prob_draw    REAL,
                prob_away    REAL,
                prob_over25  REAL,
                prob_btts    REAL,
                -- Mercado recomendado e EV
                mercado      TEXT,                     -- h2h_home, h2h_away, over25, etc.
                odd_usada    REAL,
                ev_percent   REAL,
                bookmaker    TEXT,
                -- Resultado real (preenchido depois)
                resultado    TEXT,                     -- home, draw, away
                gols_home    INTEGER,
                gols_away    INTEGER,
                acertou      INTEGER,                  -- 1=sim, 0=não, NULL=pendente
                lucro        REAL,                     -- Simulação: (odd-1) se acertou, -1 se errou
                -- Metadados
                modelo_versao TEXT DEFAULT 'v1',
                features_json TEXT,                    -- Features usadas na previsão
                created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
                resolved_at  TEXT,
                FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_fixture
                ON predictions(fixture_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_date
                ON predictions(date);
            CREATE INDEX IF NOT EXISTS idx_predictions_acertou
                ON predictions(acertou);

            -- Cache de odds do The Odds API
            CREATE TABLE IF NOT EXISTS odds_cache (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id   INTEGER,                  -- NULL se não casou com API-Football
                sport_key    TEXT NOT NULL,
                home_team    TEXT NOT NULL,
                away_team    TEXT NOT NULL,
                commence_time TEXT NOT NULL,
                bookmaker    TEXT NOT NULL,
                market       TEXT NOT NULL,             -- h2h, totals, spreads
                outcome_name TEXT NOT NULL,             -- Home, Away, Draw, Over, Under
                outcome_price REAL NOT NULL,            -- Odd decimal
                outcome_point REAL,                    -- Linha (2.5 para over/under)
                fetched_at   TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_odds_fixture
                ON odds_cache(fixture_id);
            CREATE INDEX IF NOT EXISTS idx_odds_teams
                ON odds_cache(home_team, away_team);

            -- Performance do modelo (métricas acumuladas)
            CREATE TABLE IF NOT EXISTS model_scores (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                date         TEXT NOT NULL,
                league_id    INTEGER,
                mercado      TEXT,                     -- h2h, over25, btts, etc.
                total_preds  INTEGER DEFAULT 0,
                acertos      INTEGER DEFAULT 0,
                accuracy     REAL DEFAULT 0.0,
                roi          REAL DEFAULT 0.0,         -- Retorno sobre investimento simulado
                modelo_versao TEXT DEFAULT 'v1',
                updated_at   TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Log de treinamentos do modelo
            CREATE TABLE IF NOT EXISTS train_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                date         TEXT NOT NULL,
                modelo_versao TEXT NOT NULL,
                n_samples    INTEGER,
                n_features   INTEGER,
                accuracy_train REAL,
                accuracy_test  REAL,
                metrics_json TEXT,                     -- JSON com todas as métricas
                params_json  TEXT,                     -- Hiperparâmetros usados
                created_at   TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Estratégias ativas: slices liga × mercado × faixa de confiança
            -- O AutoTuner popula esta tabela; o Scanner consulta antes de emitir tip
            CREATE TABLE IF NOT EXISTS strategies (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                mercado      TEXT NOT NULL,             -- h2h_home, over25, btts_yes, etc.
                league_id    INTEGER,                   -- NULL = todas as ligas
                conf_min     REAL NOT NULL DEFAULT 0.0, -- Confiança mínima do modelo
                conf_max     REAL NOT NULL DEFAULT 1.0, -- Confiança máxima do modelo
                accuracy     REAL NOT NULL,             -- Accuracy neste slice (teste)
                n_samples    INTEGER NOT NULL,           -- Amostras neste slice
                ev_medio     REAL DEFAULT 0.0,           -- EV médio histórico neste slice
                ativo        INTEGER NOT NULL DEFAULT 1, -- 1=emitir tips, 0=desativado
                params_json  TEXT,                       -- Hiperparâmetros que geraram este slice
                modelo_versao TEXT,
                created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(mercado, league_id, conf_min, conf_max)
            );

            -- Combos sugeridos pelo scanner (duplas/triplas)
            CREATE TABLE IF NOT EXISTS combos (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                date           TEXT NOT NULL,
                combo_type     TEXT NOT NULL,           -- dupla, tripla
                prob_composta  REAL,
                created_at     TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS combo_items (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                combo_id       INTEGER NOT NULL,
                prediction_id  INTEGER NOT NULL,
                item_order     INTEGER NOT NULL,
                fixture_id     INTEGER NOT NULL,
                mercado        TEXT NOT NULL,
                home_name      TEXT,
                away_name      TEXT,
                prob_modelo    REAL,
                FOREIGN KEY (combo_id) REFERENCES combos(id) ON DELETE CASCADE,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_combos_date
                ON combos(date);
            CREATE INDEX IF NOT EXISTS idx_combo_items_combo
                ON combo_items(combo_id);

            CREATE TABLE IF NOT EXISTS combo_live_notifications (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                combo_id       INTEGER NOT NULL,
                progress_key   TEXT NOT NULL,
                created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(combo_id, progress_key),
                FOREIGN KEY (combo_id) REFERENCES combos(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS telegram_chats (
                chat_id        INTEGER PRIMARY KEY,
                is_admin       INTEGER NOT NULL DEFAULT 0,
                username       TEXT,
                first_name     TEXT,
                created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
                last_seen_at   TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS scan_audit (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date      TEXT NOT NULL,
                fixture_id     INTEGER NOT NULL,
                league_id      INTEGER,
                home_name      TEXT,
                away_name      TEXT,
                mercado        TEXT NOT NULL,
                descricao      TEXT,
                prob_modelo    REAL,
                odd_usada      REAL,
                ev_percent     REAL,
                llm_decisao    TEXT,
                llm_confianca  REAL,
                llm_motivo     TEXT,
                approved_final INTEGER NOT NULL DEFAULT 0,
                contexto_json  TEXT,
                created_at     TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_scan_audit_date
                ON scan_audit(scan_date, llm_decisao, approved_final);

            CREATE TABLE IF NOT EXISTS scan_candidates (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date      TEXT NOT NULL,
                fixture_id      INTEGER NOT NULL,
                fixture_date    TEXT,
                league_id       INTEGER,
                home_name       TEXT,
                away_name       TEXT,
                mercado         TEXT NOT NULL,
                descricao       TEXT,
                prob_modelo     REAL,
                payload_json    TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'pending',
                release_group   TEXT DEFAULT '',
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_scan_candidates_date
                ON scan_candidates(scan_date, status, fixture_date);

            CREATE TABLE IF NOT EXISTS context_feedback (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_audit_id       INTEGER NOT NULL UNIQUE,
                fixture_id          INTEGER NOT NULL,
                league_id           INTEGER,
                mercado             TEXT NOT NULL,
                llm_decisao         TEXT,
                approved_final      INTEGER NOT NULL DEFAULT 0,
                market_won          INTEGER NOT NULL,
                context_label       TEXT NOT NULL,
                contextual_success  INTEGER NOT NULL DEFAULT 0,
                gols_home           INTEGER,
                gols_away           INTEGER,
                corners_total       INTEGER,
                weather_summary     TEXT,
                field_conditions    TEXT,
                rotation_risk       TEXT,
                motivation_context  TEXT,
                news_summary        TEXT,
                risk_flags_json     TEXT,
                llm_motivo          TEXT,
                contexto_json       TEXT,
                created_at          TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_context_feedback_fixture
                ON context_feedback(fixture_id, context_label);

            CREATE TABLE IF NOT EXISTS live_watchlist (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date        TEXT NOT NULL,
                fixture_id       INTEGER NOT NULL,
                fixture_date     TEXT,
                league_id        INTEGER,
                home_name        TEXT,
                away_name        TEXT,
                mercado          TEXT NOT NULL,
                descricao        TEXT,
                prob_modelo      REAL,
                watch_type       TEXT NOT NULL,
                status           TEXT NOT NULL DEFAULT 'active',
                note             TEXT,
                payload_json     TEXT,
                created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
                last_checked_at  TEXT,
                resolved_at      TEXT,
                UNIQUE(scan_date, fixture_id, mercado, watch_type)
            );

            CREATE INDEX IF NOT EXISTS idx_live_watchlist_status
                ON live_watchlist(status, fixture_date);
        """)
        self._aplicar_migracoes(conn)
        conn.commit()
        conn.close()

    def _aplicar_migracoes(self, conn: sqlite3.Connection):
        """Aplica migrações simples e idempotentes no schema atual."""
        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        if "prob_modelo" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN prob_modelo REAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telegram_chats (
                chat_id        INTEGER PRIMARY KEY,
                is_admin       INTEGER NOT NULL DEFAULT 0,
                username       TEXT,
                first_name     TEXT,
                created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
                last_seen_at   TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_audit (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date      TEXT NOT NULL,
                fixture_id     INTEGER NOT NULL,
                league_id      INTEGER,
                home_name      TEXT,
                away_name      TEXT,
                mercado        TEXT NOT NULL,
                descricao      TEXT,
                prob_modelo    REAL,
                odd_usada      REAL,
                ev_percent     REAL,
                llm_decisao    TEXT,
                llm_confianca  REAL,
                llm_motivo     TEXT,
                approved_final INTEGER NOT NULL DEFAULT 0,
                contexto_json  TEXT,
                created_at     TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_candidates (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date      TEXT NOT NULL,
                fixture_id      INTEGER NOT NULL,
                fixture_date    TEXT,
                league_id       INTEGER,
                home_name       TEXT,
                away_name       TEXT,
                mercado         TEXT NOT NULL,
                descricao       TEXT,
                prob_modelo     REAL,
                payload_json    TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'pending',
                release_group   TEXT DEFAULT '',
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS context_feedback (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_audit_id       INTEGER NOT NULL UNIQUE,
                fixture_id          INTEGER NOT NULL,
                league_id           INTEGER,
                mercado             TEXT NOT NULL,
                llm_decisao         TEXT,
                approved_final      INTEGER NOT NULL DEFAULT 0,
                market_won          INTEGER NOT NULL,
                context_label       TEXT NOT NULL,
                contextual_success  INTEGER NOT NULL DEFAULT 0,
                gols_home           INTEGER,
                gols_away           INTEGER,
                corners_total       INTEGER,
                weather_summary     TEXT,
                field_conditions    TEXT,
                rotation_risk       TEXT,
                motivation_context  TEXT,
                news_summary        TEXT,
                risk_flags_json     TEXT,
                llm_motivo          TEXT,
                contexto_json       TEXT,
                created_at          TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_watchlist (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date        TEXT NOT NULL,
                fixture_id       INTEGER NOT NULL,
                fixture_date     TEXT,
                league_id        INTEGER,
                home_name        TEXT,
                away_name        TEXT,
                mercado          TEXT NOT NULL,
                descricao        TEXT,
                prob_modelo      REAL,
                watch_type       TEXT NOT NULL,
                status           TEXT NOT NULL DEFAULT 'active',
                note             TEXT,
                payload_json     TEXT,
                created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
                last_checked_at  TEXT,
                resolved_at      TEXT,
                UNIQUE(scan_date, fixture_id, mercado, watch_type)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS combo_live_notifications (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                combo_id       INTEGER NOT NULL,
                progress_key   TEXT NOT NULL,
                created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(combo_id, progress_key),
                FOREIGN KEY (combo_id) REFERENCES combos(id) ON DELETE CASCADE
            )
        """)

    # ══════════════════════════════════════════════
    #  FIXTURES
    # ══════════════════════════════════════════════

    def salvar_fixture(self, f: dict):
        """
        Salva ou atualiza um fixture a partir do JSON da API-Football.
        Espera o formato: {fixture: {}, league: {}, teams: {}, goals: {}, score: {}}.
        """
        fix = f.get("fixture", {})
        league = f.get("league", {})
        teams = f.get("teams", {})
        goals = f.get("goals", {})
        score = f.get("score", {})
        ht = score.get("halftime", {})

        conn = self._conn()
        conn.execute("""
            INSERT INTO fixtures (
                fixture_id, league_id, league_name, season, round, date, timestamp,
                venue, status, home_id, home_name, away_id, away_name,
                goals_home, goals_away, score_ht_h, score_ht_a, referee, raw_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(fixture_id) DO UPDATE SET
                status=excluded.status,
                goals_home=excluded.goals_home,
                goals_away=excluded.goals_away,
                score_ht_h=excluded.score_ht_h,
                score_ht_a=excluded.score_ht_a,
                referee=excluded.referee,
                raw_json=excluded.raw_json,
                updated_at=CURRENT_TIMESTAMP
        """, (
            fix.get("id"),
            league.get("id"),
            league.get("name"),
            league.get("season"),
            league.get("round"),
            fix.get("date"),
            fix.get("timestamp"),
            (fix.get("venue") or {}).get("name"),
            (fix.get("status") or {}).get("short", "NS"),
            teams.get("home", {}).get("id"),
            teams.get("home", {}).get("name"),
            teams.get("away", {}).get("id"),
            teams.get("away", {}).get("name"),
            goals.get("home"),
            goals.get("away"),
            ht.get("home"),
            ht.get("away"),
            fix.get("referee"),
            json.dumps(f, ensure_ascii=False),
        ))
        conn.commit()
        conn.close()

    def salvar_fixtures_batch(self, fixtures: list[dict]):
        """Salva múltiplos fixtures de uma vez (transação única, mais rápido)."""
        conn = self._conn()
        for f in fixtures:
            fix = f.get("fixture", {})
            league = f.get("league", {})
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            score = f.get("score", {})
            ht = score.get("halftime", {})

            conn.execute("""
                INSERT INTO fixtures (
                    fixture_id, league_id, league_name, season, round, date, timestamp,
                    venue, status, home_id, home_name, away_id, away_name,
                    goals_home, goals_away, score_ht_h, score_ht_a, referee, raw_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(fixture_id) DO UPDATE SET
                    status=excluded.status,
                    goals_home=excluded.goals_home,
                    goals_away=excluded.goals_away,
                    score_ht_h=excluded.score_ht_h,
                    score_ht_a=excluded.score_ht_a,
                    referee=excluded.referee,
                    raw_json=excluded.raw_json,
                    updated_at=CURRENT_TIMESTAMP
            """, (
                fix.get("id"),
                league.get("id"),
                league.get("name"),
                league.get("season"),
                league.get("round"),
                fix.get("date"),
                fix.get("timestamp"),
                (fix.get("venue") or {}).get("name"),
                (fix.get("status") or {}).get("short", "NS"),
                teams.get("home", {}).get("id"),
                teams.get("home", {}).get("name"),
                teams.get("away", {}).get("id"),
                teams.get("away", {}).get("name"),
                goals.get("home"),
                goals.get("away"),
                ht.get("home"),
                ht.get("away"),
                fix.get("referee"),
                json.dumps(f, ensure_ascii=False),
            ))
        conn.commit()
        conn.close()
        return len(fixtures)

    def fixtures_por_liga(self, league_id: int, season: int) -> list[dict]:
        """Retorna todos os fixtures de uma liga/season como lista de dicts."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM fixtures WHERE league_id=? AND season=? ORDER BY date",
            (league_id, season)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def fixtures_finalizados(self, league_id: int = None, season: int = None) -> list[dict]:
        """Retorna fixtures com status FT (finalizados), filtrados opcionalmente."""
        sql = "SELECT * FROM fixtures WHERE status='FT'"
        params = []
        if league_id:
            sql += " AND league_id=?"
            params.append(league_id)
        if season:
            sql += " AND season=?"
            params.append(season)
        sql += " ORDER BY date"
        conn = self._conn()
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def fixtures_pendentes(self) -> list[dict]:
        """Retorna fixtures ainda não jogados (NS = Not Started)."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM fixtures WHERE status='NS' ORDER BY date"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def fixture_por_id(self, fixture_id: int) -> dict | None:
        """Retorna um fixture pelo ID."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM fixtures WHERE fixture_id=?", (fixture_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    # ══════════════════════════════════════════════
    #  FIXTURE STATS (estatísticas por partida)
    # ══════════════════════════════════════════════

    def salvar_fixture_stats(self, fixture_id: int, stats_list: list[dict]):
        """
        Salva estatísticas de partida (2 registros: 1 por time).
        stats_list: retorno de fixtures/statistics da API-Football.
        """
        conn = self._conn()
        for item in stats_list:
            team = item.get("team", {})
            stats = item.get("statistics", [])
            stats_dict = {s["type"]: s["value"] for s in stats}

            # Extrair campos numéricos (tratando None e '%')
            def _num(val):
                if val is None:
                    return None
                if isinstance(val, str):
                    return float(val.replace("%", "")) if val.replace(".", "").replace("%", "").isdigit() else None
                return val

            conn.execute("""
                INSERT INTO fixture_stats (
                    fixture_id, team_id, team_name, stats_json,
                    shots_total, shots_on_target, possession, passes_total,
                    passes_accuracy, fouls, corners, yellow_cards, red_cards,
                    expected_goals
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fixture_id, team_id) DO UPDATE SET
                    stats_json=excluded.stats_json,
                    shots_total=excluded.shots_total,
                    shots_on_target=excluded.shots_on_target,
                    possession=excluded.possession,
                    passes_total=excluded.passes_total,
                    passes_accuracy=excluded.passes_accuracy,
                    fouls=excluded.fouls,
                    corners=excluded.corners,
                    yellow_cards=excluded.yellow_cards,
                    red_cards=excluded.red_cards,
                    expected_goals=excluded.expected_goals
            """, (
                fixture_id,
                team.get("id"),
                team.get("name"),
                json.dumps(stats_dict, ensure_ascii=False),
                _num(stats_dict.get("Total Shots")),
                _num(stats_dict.get("Shots on Goal")),
                _num(stats_dict.get("Ball Possession")),
                _num(stats_dict.get("Total passes")),
                _num(stats_dict.get("Passes %")),
                _num(stats_dict.get("Fouls")),
                _num(stats_dict.get("Corner Kicks")),
                _num(stats_dict.get("Yellow Cards")),
                _num(stats_dict.get("Red Cards")),
                _num(stats_dict.get("expected_goals")),
            ))
        conn.commit()
        conn.close()

    def stats_partida(self, fixture_id: int) -> list[dict]:
        """Retorna stats das 2 equipes de um jogo."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM fixture_stats WHERE fixture_id=?", (fixture_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ══════════════════════════════════════════════
    #  FIXTURE EVENTS (gols, cartões, etc.)
    # ══════════════════════════════════════════════

    def salvar_eventos(self, fixture_id: int, events: list[dict]):
        """Salva eventos de uma partida."""
        conn = self._conn()
        # Limpa eventos antigos antes de reinserir
        conn.execute("DELETE FROM fixture_events WHERE fixture_id=?", (fixture_id,))
        for ev in events:
            conn.execute("""
                INSERT INTO fixture_events (
                    fixture_id, team_id, player_name, event_type,
                    event_detail, minute, extra_minute, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fixture_id,
                (ev.get("team") or {}).get("id"),
                (ev.get("player") or {}).get("name"),
                ev.get("type"),
                ev.get("detail"),
                (ev.get("time") or {}).get("elapsed"),
                (ev.get("time") or {}).get("extra"),
                json.dumps(ev, ensure_ascii=False),
            ))
        conn.commit()
        conn.close()

    # ══════════════════════════════════════════════
    #  TEAM STATS (agregadas por liga/season)
    # ══════════════════════════════════════════════

    def salvar_team_stats(self, team_id: int, league_id: int, season: int,
                          team_name: str, stats: dict):
        """Salva estatísticas agregadas de um time em uma liga/season."""
        fix = stats.get("fixtures", {})
        goals = stats.get("goals", {})

        conn = self._conn()
        conn.execute("""
            INSERT INTO team_stats (
                team_id, league_id, season, team_name, form, stats_json,
                played, wins, draws, losses, goals_for, goals_against,
                goals_for_avg, goals_against_avg, clean_sheets, failed_to_score,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(team_id, league_id, season) DO UPDATE SET
                form=excluded.form,
                stats_json=excluded.stats_json,
                played=excluded.played,
                wins=excluded.wins,
                draws=excluded.draws,
                losses=excluded.losses,
                goals_for=excluded.goals_for,
                goals_against=excluded.goals_against,
                goals_for_avg=excluded.goals_for_avg,
                goals_against_avg=excluded.goals_against_avg,
                clean_sheets=excluded.clean_sheets,
                failed_to_score=excluded.failed_to_score,
                updated_at=CURRENT_TIMESTAMP
        """, (
            team_id, league_id, season, team_name,
            stats.get("form"),
            json.dumps(stats, ensure_ascii=False),
            (fix.get("played") or {}).get("total"),
            (fix.get("wins") or {}).get("total"),
            (fix.get("draws") or {}).get("total"),
            (fix.get("loses") or {}).get("total"),
            (goals.get("for", {}).get("total") or {}).get("total"),
            (goals.get("against", {}).get("total") or {}).get("total"),
            float((goals.get("for", {}).get("average") or {}).get("total") or 0),
            float((goals.get("against", {}).get("average") or {}).get("total") or 0),
            (stats.get("clean_sheet") or {}).get("total"),
            (stats.get("failed_to_score") or {}).get("total"),
        ))
        conn.commit()
        conn.close()

    # ══════════════════════════════════════════════
    #  PREDICTIONS (registro e resolução)
    # ══════════════════════════════════════════════

    def salvar_prediction(self, pred: dict):
        """Salva uma previsão do modelo.

        Usa INSERT OR IGNORE para evitar duplicatas quando o
        scanner é executado mais de uma vez no mesmo dia.
        Depende do índice UNIQUE(fixture_id, mercado) na tabela.
        """
        conn = self._conn()
        # Garantir UNIQUE constraint existe (idempotente)
        try:
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_pred_fixture_mercado
                ON predictions(fixture_id, mercado)
            """)
        except Exception:
            pass  # Já existe ou banco read-only
        conn.execute("""
            INSERT OR IGNORE INTO predictions (
                fixture_id, date, league_id, home_name, away_name,
                prob_home, prob_draw, prob_away, prob_over25, prob_btts, prob_modelo,
                mercado, odd_usada, ev_percent, bookmaker,
                modelo_versao, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred["fixture_id"], pred["date"], pred.get("league_id"),
            pred.get("home_name"), pred.get("away_name"),
            pred.get("prob_home"), pred.get("prob_draw"), pred.get("prob_away"),
            pred.get("prob_over25"), pred.get("prob_btts"), pred.get("prob_modelo"),
            pred.get("mercado"), pred.get("odd_usada"), pred.get("ev_percent"),
            pred.get("bookmaker"), pred.get("modelo_versao", "v1"),
            json.dumps(pred.get("features", {}), ensure_ascii=False),
        ))
        conn.commit()
        conn.close()

    def buscar_prediction(self, fixture_id: int, mercado: str) -> dict | None:
        """Busca uma previsão pelo fixture_id + mercado. Retorna dict ou None."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM predictions WHERE fixture_id = ? AND mercado = ?",
            (fixture_id, mercado)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def limpar_lote_scan(self, data: str):
        """
        Remove o lote aberto de scan de uma data antes de salvar um novo.

        Mantém previsões já resolvidas para não apagar histórico após os jogos.
        Remove combos do dia inteiro, porque combos são sempre regenerados a cada scan.
        """
        conn = self._conn()
        conn.execute(
            "DELETE FROM combos WHERE date LIKE ?",
            (f"{data}%",),
        )
        conn.execute(
            "DELETE FROM predictions WHERE date LIKE ? AND acertou IS NULL",
            (f"{data}%",),
        )
        conn.execute(
            "DELETE FROM scan_audit WHERE scan_date = ?",
            (data,),
        )
        conn.execute(
            "DELETE FROM scan_candidates WHERE scan_date = ?",
            (data,),
        )
        conn.execute(
            "DELETE FROM live_watchlist WHERE scan_date = ?",
            (data,),
        )
        conn.commit()
        conn.close()

    def limpar_scan_candidates(self, data: str):
        """Remove apenas o radar pendente do dia, sem apagar tip/live já em andamento."""
        conn = self._conn()
        conn.execute(
            "DELETE FROM scan_candidates WHERE scan_date = ?",
            (data,),
        )
        conn.commit()
        conn.close()

    def salvar_scan_candidates(self, data: str, tips: list[dict]):
        """Persiste candidatos pré-selecionados do scan da manhã."""
        if not tips:
            return
        conn = self._conn()
        for tip in tips:
            conn.execute("""
                INSERT INTO scan_candidates (
                    scan_date, fixture_id, fixture_date, league_id, home_name, away_name,
                    mercado, descricao, prob_modelo, payload_json, status, release_group
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data,
                tip.get("fixture_id"),
                tip.get("date"),
                tip.get("league_id"),
                tip.get("home_name"),
                tip.get("away_name"),
                tip.get("mercado"),
                tip.get("descricao"),
                tip.get("prob_modelo"),
                json.dumps(tip, ensure_ascii=False),
                tip.get("candidate_status", "pending"),
                tip.get("release_group", ""),
            ))
        conn.commit()
        conn.close()

    def candidatos_por_data(self, data: str, status: str = None) -> list[dict]:
        """Retorna candidatos pré-selecionados do dia."""
        conn = self._conn()
        sql = """
            SELECT *
            FROM scan_candidates
            WHERE scan_date = ?
        """
        params = [data]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY fixture_date ASC, prob_modelo DESC, id ASC"
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.get("payload_json") or "{}")
            except json.JSONDecodeError:
                item["payload"] = {}
            out.append(item)
        return out

    def atualizar_status_candidatos(self, candidate_ids: list[int], status: str):
        """Atualiza status de um lote de candidatos."""
        if not candidate_ids:
            return
        conn = self._conn()
        marks = ",".join("?" for _ in candidate_ids)
        conn.execute(
            f"UPDATE scan_candidates SET status = ? WHERE id IN ({marks})",
            [status, *candidate_ids],
        )
        conn.commit()
        conn.close()

    def salvar_scan_audit(self, data: str, tips: list[dict]):
        """Persiste a decisao do LLM para aprovadas e rejeitadas de um scan."""
        if not tips:
            return

        conn = self._conn()
        for tip in tips:
            llm = tip.get("llm_validacao") or {}
            contexto = tip.get("llm_contexto") or {}
            conn.execute("""
                INSERT INTO scan_audit (
                    scan_date, fixture_id, league_id, home_name, away_name,
                    mercado, descricao, prob_modelo, odd_usada, ev_percent,
                    llm_decisao, llm_confianca, llm_motivo, approved_final, contexto_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data,
                tip.get("fixture_id"),
                tip.get("league_id"),
                tip.get("home_name"),
                tip.get("away_name"),
                tip.get("mercado"),
                tip.get("descricao"),
                tip.get("prob_modelo"),
                tip.get("odd_pinnacle") or tip.get("odd_usada"),
                tip.get("ev_percent"),
                llm.get("decisao"),
                llm.get("confianca"),
                llm.get("motivo"),
                1 if tip.get("approved_final") else 0,
                json.dumps(contexto, ensure_ascii=False),
            ))
        conn.commit()
        conn.close()

    def salvar_live_watchlist(self, data: str, itens: list[dict]):
        """Persistir jogos/mercados que merecem acompanhamento live."""
        if not itens:
            return

        conn = self._conn()
        for item in itens:
            conn.execute("""
                INSERT OR REPLACE INTO live_watchlist (
                    scan_date, fixture_id, fixture_date, league_id,
                    home_name, away_name, mercado, descricao, prob_modelo,
                    watch_type, status, note, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data,
                item.get("fixture_id"),
                item.get("date") or item.get("fixture_date"),
                item.get("league_id"),
                item.get("home_name"),
                item.get("away_name"),
                item.get("mercado"),
                item.get("descricao"),
                item.get("prob_modelo"),
                item.get("watch_type", "approved_prelive"),
                item.get("status", "active"),
                item.get("note"),
                json.dumps(item.get("payload") or item, ensure_ascii=False),
            ))
        conn.commit()
        conn.close()

    def live_watch_items(self, dates: list[str] | None = None, status: str | None = "active") -> list[dict]:
        """Retorna itens monitorados para acompanhamento live."""
        conn = self._conn()
        sql = """
            SELECT *
            FROM live_watchlist
            WHERE 1 = 1
        """
        params = []
        if dates:
            marks = ",".join("?" for _ in dates)
            sql += f" AND scan_date IN ({marks})"
            params.extend(dates)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY fixture_date ASC, prob_modelo DESC, id ASC"
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.get("payload_json") or "{}")
            except json.JSONDecodeError:
                item["payload"] = {}
            out.append(item)
        return out

    def atualizar_status_live_watchlist(self, item_ids: list[int], status: str):
        """Atualiza status de itens da watchlist live."""
        if not item_ids:
            return
        conn = self._conn()
        marks = ",".join("?" for _ in item_ids)
        conn.execute(
            f"""
            UPDATE live_watchlist
            SET status = ?, last_checked_at = CURRENT_TIMESTAMP,
                resolved_at = CASE WHEN ? = 'resolved' THEN CURRENT_TIMESTAMP ELSE resolved_at END
            WHERE id IN ({marks})
            """,
            [status, status, *item_ids],
        )
        conn.commit()
        conn.close()

    def tocar_live_watchlist(self, item_ids: list[int]):
        """Marca itens da watchlist como checados neste ciclo."""
        if not item_ids:
            return
        conn = self._conn()
        marks = ",".join("?" for _ in item_ids)
        conn.execute(
            f"UPDATE live_watchlist SET last_checked_at = CURRENT_TIMESTAMP WHERE id IN ({marks})",
            item_ids,
        )
        conn.commit()
        conn.close()

    def atualizar_live_watch_item(
        self,
        item_id: int,
        *,
        status: str | None = None,
        note: str | None = None,
        payload: dict | None = None,
    ):
        """Atualiza campos mutáveis de um item da watchlist live."""
        sets = ["last_checked_at = CURRENT_TIMESTAMP"]
        params = []

        if status is not None:
            sets.append("status = ?")
            params.append(status)
            if status == "resolved":
                sets.append("resolved_at = CURRENT_TIMESTAMP")
        if note is not None:
            sets.append("note = ?")
            params.append(note)
        if payload is not None:
            sets.append("payload_json = ?")
            params.append(json.dumps(payload, ensure_ascii=False))

        params.append(item_id)
        conn = self._conn()
        conn.execute(
            f"UPDATE live_watchlist SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        conn.commit()
        conn.close()

    def scan_audit_por_data(self, data: str, decisao: str = None) -> list[dict]:
        """Retorna a trilha do LLM para um scan especifico."""
        conn = self._conn()
        sql = """
            SELECT *
            FROM scan_audit
            WHERE scan_date = ?
        """
        params = [data]
        if decisao:
            sql += " AND llm_decisao = ?"
            params.append(decisao)
        sql += """
            ORDER BY
                league_id ASC,
                home_name ASC,
                prob_modelo DESC,
                id ASC
        """
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def scan_audit_por_fixture(self, fixture_id: int) -> list[dict]:
        """Retorna auditoria do scan para um fixture específico."""
        conn = self._conn()
        rows = conn.execute("""
            SELECT *
            FROM scan_audit
            WHERE fixture_id = ?
            ORDER BY created_at ASC, id ASC
        """, (fixture_id,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def scan_audit_fixtures_sem_feedback(self) -> list[int]:
        """Lista fixtures auditados que já podem gerar label contextual e ainda não geraram."""
        conn = self._conn()
        rows = conn.execute("""
            SELECT DISTINCT sa.fixture_id
            FROM scan_audit sa
            JOIN fixtures f
              ON f.fixture_id = sa.fixture_id
            LEFT JOIN context_feedback cf
              ON cf.scan_audit_id = sa.id
            WHERE f.status = 'FT'
              AND f.goals_home IS NOT NULL
              AND f.goals_away IS NOT NULL
              AND cf.id IS NULL
            ORDER BY sa.fixture_id ASC
        """).fetchall()
        conn.close()
        return [int(r["fixture_id"]) for r in rows]

    def salvar_context_feedback(self, feedbacks: list[dict]):
        """Salva labels de sucesso/erro contextual derivados do scan_audit."""
        if not feedbacks:
            return
        conn = self._conn()
        for item in feedbacks:
            conn.execute("""
                INSERT OR IGNORE INTO context_feedback (
                    scan_audit_id, fixture_id, league_id, mercado,
                    llm_decisao, approved_final, market_won, context_label,
                    contextual_success, gols_home, gols_away, corners_total,
                    weather_summary, field_conditions, rotation_risk,
                    motivation_context, news_summary, risk_flags_json,
                    llm_motivo, contexto_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item["scan_audit_id"],
                item["fixture_id"],
                item.get("league_id"),
                item["mercado"],
                item.get("llm_decisao"),
                1 if item.get("approved_final") else 0,
                1 if item.get("market_won") else 0,
                item["context_label"],
                1 if item.get("contextual_success") else 0,
                item.get("gols_home"),
                item.get("gols_away"),
                item.get("corners_total"),
                item.get("weather_summary"),
                item.get("field_conditions"),
                item.get("rotation_risk"),
                item.get("motivation_context"),
                item.get("news_summary"),
                json.dumps(item.get("risk_flags", []), ensure_ascii=False),
                item.get("llm_motivo"),
                json.dumps(item.get("contexto_json", {}), ensure_ascii=False),
            ))
        conn.commit()
        conn.close()

    def context_feedback_resumo(self) -> list[dict]:
        """Resume labels contextuais já consolidados."""
        conn = self._conn()
        rows = conn.execute("""
            SELECT
                context_label,
                COUNT(*) AS total,
                SUM(contextual_success) AS sucessos
            FROM context_feedback
            GROUP BY context_label
            ORDER BY total DESC, context_label ASC
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def salvar_combo(self, combo: dict):
        """Salva um combo e seus itens vinculados às predictions já persistidas."""
        itens = combo.get("tips", [])
        if not itens:
            return

        conn = self._conn()
        cur = conn.execute("""
            INSERT INTO combos (date, combo_type, prob_composta)
            VALUES (?, ?, ?)
        """, (
            combo.get("date"),
            combo.get("tipo", "dupla"),
            combo.get("prob_composta"),
        ))
        combo_id = cur.lastrowid

        for idx, item in enumerate(itens, start=1):
            pred = conn.execute(
                "SELECT id FROM predictions WHERE fixture_id = ? AND mercado = ?",
                (item.get("fixture_id"), item.get("mercado")),
            ).fetchone()
            if not pred:
                continue

            conn.execute("""
                INSERT INTO combo_items (
                    combo_id, prediction_id, item_order, fixture_id, mercado,
                    home_name, away_name, prob_modelo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                combo_id,
                pred["id"],
                idx,
                item.get("fixture_id"),
                item.get("mercado"),
                item.get("home_name"),
                item.get("away_name"),
                item.get("prob_modelo"),
            ))

        conn.commit()
        conn.close()

    def combos_por_data(self, data: str) -> list[dict]:
        """Retorna combos e itens associados para uma data ISO."""
        conn = self._conn()
        combos = conn.execute("""
            SELECT * FROM combos
            WHERE date LIKE ?
            ORDER BY created_at ASC, id ASC
        """, (f"{data}%",)).fetchall()

        resultado = []
        for combo in combos:
            itens = conn.execute("""
                SELECT
                    ci.*,
                    p.acertou,
                    p.gols_home,
                    p.gols_away,
                    p.odd_usada,
                    p.lucro
                FROM combo_items ci
                JOIN predictions p ON p.id = ci.prediction_id
                WHERE ci.combo_id = ?
                ORDER BY ci.item_order ASC
            """, (combo["id"],)).fetchall()
            payload = dict(combo)
            payload["items"] = [dict(i) for i in itens]
            resultado.append(payload)

        conn.close()
        return resultado

    def combo_live_notification_exists(self, combo_id: int, progress_key: str) -> bool:
        conn = self._conn()
        row = conn.execute(
            "SELECT 1 FROM combo_live_notifications WHERE combo_id = ? AND progress_key = ? LIMIT 1",
            (combo_id, progress_key),
        ).fetchone()
        conn.close()
        return row is not None

    def salvar_combo_live_notification(self, combo_id: int, progress_key: str):
        conn = self._conn()
        conn.execute(
            """
            INSERT OR IGNORE INTO combo_live_notifications (combo_id, progress_key)
            VALUES (?, ?)
            """,
            (combo_id, progress_key),
        )
        conn.commit()
        conn.close()

    def atualizar_odd_manual(self, fixture_id: int, mercado: str,
                             odd: float, ev_percent: float,
                             bookmaker: str = "manual"):
        """Atualiza odd e EV de uma previsão (input manual do usuário)."""
        conn = self._conn()
        conn.execute("""
            UPDATE predictions
            SET odd_usada = ?, ev_percent = ?, bookmaker = ?
            WHERE fixture_id = ? AND mercado = ?
        """, (odd, ev_percent, bookmaker, fixture_id, mercado))
        conn.commit()
        conn.close()

    def resolver_prediction(self, fixture_id: int, resultado: str,
                            gols_home: int, gols_away: int):
        """
        Preenche resultado real de uma previsão.
        resultado: 'home', 'draw' ou 'away'.
        """
        conn = self._conn()
        preds = conn.execute(
            "SELECT id, mercado, odd_usada FROM predictions WHERE fixture_id=? AND acertou IS NULL",
            (fixture_id,)
        ).fetchall()

        for p in preds:
            mercado = p["mercado"]
            odd = p["odd_usada"] or 1.0
            total_gols = gols_home + gols_away

            # Determinar se acertou com base no mercado.
            # Cobre TODOS os mercados que o scanner pode gerar,
            # incluindo over/under 1.5 e 3.5 e resultado do 1º tempo.
            acertou = 0
            if mercado == "h2h_home" and resultado == "home":
                acertou = 1
            elif mercado == "h2h_draw" and resultado == "draw":
                acertou = 1
            elif mercado == "h2h_away" and resultado == "away":
                acertou = 1
            elif mercado == "over15" and total_gols > 1:
                acertou = 1
            elif mercado == "under15" and total_gols < 2:
                acertou = 1
            elif mercado == "over25" and total_gols > 2:
                acertou = 1
            elif mercado == "under25" and total_gols < 3:
                acertou = 1
            elif mercado == "over35" and total_gols > 3:
                acertou = 1
            elif mercado == "under35" and total_gols < 4:
                acertou = 1
            elif mercado == "btts_yes" and gols_home > 0 and gols_away > 0:
                acertou = 1
            elif mercado == "btts_no" and (gols_home == 0 or gols_away == 0):
                acertou = 1
            elif mercado == "ht_home" and resultado == "home":
                # Aproximação: sem dados de 1T, usamos resultado FT
                acertou = 1
            elif mercado == "ht_draw" and resultado == "draw":
                acertou = 1
            elif mercado == "ht_away" and resultado == "away":
                acertou = 1

            lucro = (odd - 1) if acertou else -1.0

            conn.execute("""
                UPDATE predictions SET
                    resultado=?, gols_home=?, gols_away=?,
                    acertou=?, lucro=?, resolved_at=CURRENT_TIMESTAMP
                WHERE id=?
            """, (resultado, gols_home, gols_away, acertou, lucro, p["id"]))

        conn.commit()
        conn.close()
        return len(preds)

    def predictions_pendentes(self) -> list[dict]:
        """Retorna previsões que ainda não foram resolvidas."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM predictions WHERE acertou IS NULL ORDER BY date"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def predictions_por_data(self, data: str) -> list[dict]:
        """Retorna previsões salvas em uma data ISO, com nome da liga."""
        conn = self._conn()
        rows = conn.execute("""
            SELECT
                p.*,
                COALESCE(f.league_name, 'Liga ' || p.league_id) AS league_name,
                COALESCE(f.date, p.date) AS fixture_date
            FROM predictions p
            LEFT JOIN fixtures f ON f.fixture_id = p.fixture_id
            WHERE p.date LIKE ?
            ORDER BY
                COALESCE(f.league_name, 'Liga ' || p.league_id) ASC,
                COALESCE(f.date, p.date) ASC,
                p.fixture_id ASC,
                p.prob_modelo DESC,
                p.id ASC
        """, (f"{data}%",)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def metricas_modelo(self, league_id: int = None, mercado: str = None,
                         modelo_versao: str = None) -> dict:
        """
        Calcula métricas de performance do modelo.

        Parâmetros opcionais:
          league_id: filtrar por liga específica
          mercado: filtrar por mercado (h2h_home, over25, etc.)
          modelo_versao: filtrar apenas previsões feitas por esta versão do modelo.
                         Se informado, ignora previsões de versões anteriores.
        """
        sql = "SELECT * FROM predictions WHERE acertou IS NOT NULL"
        params = []
        if modelo_versao:
            sql += " AND modelo_versao=?"
            params.append(modelo_versao)
        if league_id:
            sql += " AND league_id=?"
            params.append(league_id)
        if mercado:
            sql += " AND mercado=?"
            params.append(mercado)

        conn = self._conn()
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        if not rows:
            return {"total": 0, "acertos": 0, "accuracy": 0, "roi": 0, "lucro_total": 0}

        total = len(rows)
        acertos = sum(1 for r in rows if r["acertou"] == 1)
        lucro_total = sum(r["lucro"] or 0 for r in rows)

        return {
            "total": total,
            "acertos": acertos,
            "accuracy": round(acertos / total * 100, 1) if total else 0,
            "roi": round(lucro_total / total * 100, 1) if total else 0,
            "lucro_total": round(lucro_total, 2),
        }

    def metricas_por_mercado_liga(self, min_amostras: int = 3,
                                   modelo_versao: str = None) -> list[dict]:
        """
        Retorna métricas agrupadas por mercado × liga.

        Faz JOIN com fixtures para obter league_name.
        Filtra combinações com pelo menos `min_amostras` previsões resolvidas.

        Parâmetros:
          min_amostras: número mínimo de previsões resolvidas por grupo
          modelo_versao: se informado, filtra apenas previsões desta versão

        Retorna lista de dicts ordenada por mercado, depois accuracy desc.
        """
        # Montar cláusula WHERE dinâmica para filtro de modelo
        where_extra = ""
        params = []
        if modelo_versao:
            where_extra = " AND p.modelo_versao = ?"
            params.append(modelo_versao)
        params.append(min_amostras)

        conn = self._conn()
        rows = conn.execute(f"""
            SELECT
                p.mercado,
                p.league_id,
                COALESCE(f.league_name, 'Liga ' || p.league_id) AS league_name,
                COUNT(*)                                        AS total,
                SUM(CASE WHEN p.acertou = 1 THEN 1 ELSE 0 END) AS acertos,
                SUM(COALESCE(p.lucro, 0))                       AS lucro
            FROM predictions p
            LEFT JOIN fixtures f ON f.fixture_id = p.fixture_id
            WHERE p.acertou IS NOT NULL
              AND p.mercado IS NOT NULL
              AND p.league_id IS NOT NULL
              {where_extra}
            GROUP BY p.mercado, p.league_id
            HAVING COUNT(*) >= ?
            ORDER BY p.mercado, acertos * 1.0 / COUNT(*) DESC
        """, params).fetchall()
        conn.close()

        result = []
        for r in rows:
            total = r["total"]
            acertos = r["acertos"]
            lucro = r["lucro"] or 0
            result.append({
                "mercado": r["mercado"],
                "league_id": r["league_id"],
                "league_name": r["league_name"],
                "total": total,
                "acertos": acertos,
                "accuracy": round(acertos / total * 100, 1) if total else 0,
                "roi": round(lucro / total * 100, 1) if total else 0,
                "lucro": round(lucro, 2),
            })
        return result

    # ══════════════════════════════════════════════
    #  ODDS CACHE
    # ══════════════════════════════════════════════

    def salvar_odds(self, odds_data: list[dict]):
        """Salva odds do The Odds API no cache."""
        conn = self._conn()
        for game in odds_data:
            for bk in game.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    for outcome in mkt.get("outcomes", []):
                        conn.execute("""
                            INSERT INTO odds_cache (
                                sport_key, home_team, away_team, commence_time,
                                bookmaker, market, outcome_name, outcome_price, outcome_point
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            game.get("sport_key"),
                            game.get("home_team"),
                            game.get("away_team"),
                            game.get("commence_time"),
                            bk.get("key"),
                            mkt.get("key"),
                            outcome.get("name"),
                            outcome.get("price"),
                            outcome.get("point"),
                        ))
        conn.commit()
        conn.close()

    def odds_por_jogo(self, home_team: str, away_team: str) -> list[dict]:
        """Busca odds cacheadas por nomes de times (busca parcial)."""
        conn = self._conn()
        rows = conn.execute("""
            SELECT * FROM odds_cache
            WHERE home_team LIKE ? AND away_team LIKE ?
            ORDER BY fetched_at DESC
        """, (f"%{home_team}%", f"%{away_team}%")).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ══════════════════════════════════════════════
    #  TRAIN LOG
    # ══════════════════════════════════════════════

    def salvar_treino(self, versao: str, n_samples: int, n_features: int,
                      acc_train: float, acc_test: float, metrics: dict, params: dict):
        """Registra um treinamento do modelo."""
        conn = self._conn()
        conn.execute("""
            INSERT INTO train_log (
                date, modelo_versao, n_samples, n_features,
                accuracy_train, accuracy_test, metrics_json, params_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            versao, n_samples, n_features,
            acc_train, acc_test,
            json.dumps(metrics, ensure_ascii=False),
            json.dumps(params, ensure_ascii=False),
        ))
        conn.commit()
        conn.close()

    def ultimo_treino(self) -> dict | None:
        """Retorna info do último treinamento."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM train_log ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    # ══════════════════════════════════════════════
    #  CONTADORES (visão geral)
    # ══════════════════════════════════════════════

    def resumo(self) -> dict:
        """Retorna resumo do banco (contagens de cada tabela)."""
        conn = self._conn()
        r = {}
        for tabela in ["fixtures", "fixture_stats", "fixture_events",
                       "team_stats", "predictions", "odds_cache", "train_log",
                       "telegram_chats"]:
            row = conn.execute(f"SELECT COUNT(*) as n FROM {tabela}").fetchone()
            r[tabela] = row["n"]
        # Fixtures finalizados
        row = conn.execute("SELECT COUNT(*) as n FROM fixtures WHERE status='FT'").fetchone()
        r["fixtures_ft"] = row["n"]
        # Fixtures com stats
        row = conn.execute("SELECT COUNT(DISTINCT fixture_id) as n FROM fixture_stats").fetchone()
        r["fixtures_com_stats"] = row["n"]
        # Estratégias ativas
        try:
            row = conn.execute("SELECT COUNT(*) as n FROM strategies WHERE ativo=1").fetchone()
            r["strategies_ativas"] = row["n"]
            row = conn.execute("SELECT COUNT(*) as n FROM strategies").fetchone()
            r["strategies"] = row["n"]
        except Exception:
            r["strategies_ativas"] = 0
            r["strategies"] = 0
        conn.close()
        return r

    # ══════════════════════════════════════════════
    #  STRATEGIES (AutoTuner)
    # ══════════════════════════════════════════════

    def salvar_telegram_chat(self, chat_id: int, is_admin: bool = False,
                             username: str = None, first_name: str = None):
        """Cria ou atualiza um chat conhecido do Telegram."""
        conn = self._conn()
        conn.execute("""
            INSERT INTO telegram_chats (
                chat_id, is_admin, username, first_name, last_seen_at
            ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chat_id) DO UPDATE SET
                is_admin = CASE
                    WHEN telegram_chats.is_admin = 1 OR excluded.is_admin = 1 THEN 1
                    ELSE 0
                END,
                username = COALESCE(excluded.username, telegram_chats.username),
                first_name = COALESCE(excluded.first_name, telegram_chats.first_name),
                last_seen_at = CURRENT_TIMESTAMP
        """, (
            int(chat_id),
            1 if is_admin else 0,
            username,
            first_name,
        ))
        conn.commit()
        conn.close()

    def telegram_chat_ids(self, apenas_admin: bool = False) -> list[int]:
        """Lista chat_ids conhecidos, opcionalmente filtrando apenas admins."""
        conn = self._conn()
        sql = "SELECT chat_id FROM telegram_chats"
        if apenas_admin:
            sql += " WHERE is_admin = 1"
        sql += " ORDER BY chat_id"
        rows = conn.execute(sql).fetchall()
        conn.close()
        return [int(r["chat_id"]) for r in rows]

    def salvar_strategies(self, strategies: list[dict], replace: bool = True,
                          league_ids: list[int] = None):
        """
        Salva lista de estratégias (slices) geradas pelo AutoTuner.
        Pode substituir tudo ou apenas ligas específicas.
        """
        conn = self._conn()
        if replace:
            conn.execute("DELETE FROM strategies")
        elif league_ids:
            placeholders = ",".join("?" for _ in league_ids)
            conn.execute(
                f"DELETE FROM strategies WHERE league_id IN ({placeholders})",
                tuple(league_ids),
            )
        for s in strategies:
            conn.execute("""
                INSERT INTO strategies (
                    mercado, league_id, conf_min, conf_max,
                    accuracy, n_samples, ev_medio, ativo,
                    params_json, modelo_versao
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                s["mercado"], s.get("league_id"),
                s["conf_min"], s["conf_max"],
                s["accuracy"], s["n_samples"],
                s.get("ev_medio", 0), s.get("ativo", 1),
                json.dumps(s.get("params", {}), ensure_ascii=False),
                s.get("modelo_versao", ""),
            ))
        conn.commit()
        conn.close()

    def salvar_strategies_por_slice(self, strategies: list[dict]):
        """
        Substitui strategies apenas dos slices (liga x mercado) informados.

        Mantém intactos os demais mercados da liga. Útil para discovery semanal,
        onde alguns mercados acham regra nova e outros não.
        """
        if not strategies:
            return

        conn = self._conn()
        slices = sorted({
            (s["mercado"], s.get("league_id"))
            for s in strategies
        })

        for mercado, league_id in slices:
            conn.execute(
                "DELETE FROM strategies WHERE mercado = ? AND league_id = ?",
                (mercado, league_id),
            )

        for s in strategies:
            conn.execute("""
                INSERT INTO strategies (
                    mercado, league_id, conf_min, conf_max,
                    accuracy, n_samples, ev_medio, ativo,
                    params_json, modelo_versao
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                s["mercado"], s.get("league_id"),
                s["conf_min"], s["conf_max"],
                s["accuracy"], s["n_samples"],
                s.get("ev_medio", 0), s.get("ativo", 1),
                json.dumps(s.get("params", {}), ensure_ascii=False),
                s.get("modelo_versao", ""),
            ))
        conn.commit()
        conn.close()

    def desativar_strategia_slice(self, league_id: int, mercado: str) -> int:
        """Desativa todas as strategies de um slice liga × mercado."""
        conn = self._conn()
        cur = conn.execute("""
            UPDATE strategies
            SET ativo = 0
            WHERE league_id = ? AND mercado = ? AND ativo = 1
        """, (league_id, mercado))
        conn.commit()
        alteradas = cur.rowcount
        conn.close()
        return alteradas

    def strategies_ativas(self) -> list[dict]:
        """Retorna todas as estratégias ativas para uso pelo Scanner."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM strategies WHERE ativo = 1 ORDER BY accuracy DESC"
            ).fetchall()
        except Exception:
            rows = []
        conn.close()
        return [dict(r) for r in rows]

    def slices_degradados(self, modelo_versao: str = None,
                          min_amostras: int = 5,
                          roi_threshold: float = -15.0,
                          acc_threshold: float = 35.0) -> list[dict]:
        """
        Retorna slices liga × mercado com performance ruim.

        Usa apenas previsões resolvidas e já agregadas por mercado × liga.
        """
        dados = self.metricas_por_mercado_liga(
            min_amostras=min_amostras,
            modelo_versao=modelo_versao,
        )
        ruins = []
        for item in dados:
            if item["roi"] < roi_threshold or item["accuracy"] < acc_threshold:
                ruins.append(item)
        return ruins

    def strategies_resumo(self) -> dict:
        """Retorna resumo das estratégias para exibição."""
        conn = self._conn()
        try:
            total = conn.execute("SELECT COUNT(*) as n FROM strategies").fetchone()["n"]
            ativas = conn.execute("SELECT COUNT(*) as n FROM strategies WHERE ativo=1").fetchone()["n"]
            if ativas > 0:
                avg = conn.execute(
                    "SELECT AVG(accuracy) as avg_acc FROM strategies WHERE ativo=1"
                ).fetchone()["avg_acc"]
                best = conn.execute(
                    "SELECT mercado, league_id, accuracy, n_samples "
                    "FROM strategies WHERE ativo=1 ORDER BY accuracy DESC LIMIT 5"
                ).fetchall()
            else:
                avg = 0
                best = []
        except Exception:
            total, ativas, avg, best = 0, 0, 0, []
        conn.close()
        return {
            "total": total,
            "ativas": ativas,
            "accuracy_media": round(avg, 4) if avg else 0,
            "top5": [dict(r) for r in best],
        }
