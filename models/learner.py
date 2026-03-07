"""
Learner — módulo de feedback e aprendizado contínuo.

Responsável por:
  1. Coletar resultados reais dos jogos que tiveram previsão
  2. Comparar previsão vs realidade
  3. Atualizar métricas de performance do modelo
  4. Gerar relatórios de evolução

Este módulo fecha o loop de aprendizado:
  Scanner → Predictor → Odds → EV → [JOGO ACONTECE] → Learner → Retreino

Uso:
  from models.learner import Learner
  learner = Learner(db)
  resolvidos = learner.resolver_pendentes()
  relatorio = learner.relatorio_diario()
"""

from datetime import datetime, timedelta
from data.database import Database
from services.apifootball import raw_request
from config import (
    DEGRADATION_WINDOW_DAYS, DEGRADATION_ACC_MIN,
    ROI_PAUSE_THRESHOLD, ROI_PAUSE_MIN_BETS,
)


class Learner:
    """Fecha o ciclo de aprendizado: resultados → métricas → retreino."""

    def __init__(self, db: Database):
        self.db = db

    def resolver_pendentes(self) -> dict:
        """
        Busca resultados reais de jogos que tiveram previsão e resolve.

        Fluxo:
          1. Listar predictions com acertou=NULL
          2. Para cada uma, buscar resultado real no SQLite ou na API
          3. Resolver (preencher acertou, lucro)

        Retorna resumo: {total_pendentes, resolvidos, acertos, erros}.
        """
        pendentes = self.db.predictions_pendentes()
        if not pendentes:
            return {"total_pendentes": 0, "resolvidos": 0}

        resolvidos = 0
        acertos = 0
        erros = 0

        for pred in pendentes:
            fixture_id = pred["fixture_id"]

            # Tentar pegar resultado do SQLite primeiro
            fix = self.db.fixture_por_id(fixture_id)

            if fix and fix["status"] == "FT" and fix["goals_home"] is not None:
                # Resultado já está no banco
                gh = fix["goals_home"]
                ga = fix["goals_away"]
            else:
                # Buscar resultado na API
                r = raw_request("fixtures", {"id": fixture_id})
                resp = r.get("response", [])
                if not resp:
                    continue

                game = resp[0]
                status = game.get("fixture", {}).get("status", {}).get("short", "NS")

                if status != "FT":
                    continue  # Jogo ainda não terminou

                # Atualizar fixture no banco
                self.db.salvar_fixture(game)

                gh = game.get("goals", {}).get("home")
                ga = game.get("goals", {}).get("away")

                if gh is None or ga is None:
                    continue

            # Determinar resultado
            if gh > ga:
                resultado = "home"
            elif gh == ga:
                resultado = "draw"
            else:
                resultado = "away"

            # Resolver a previsão
            n = self.db.resolver_prediction(fixture_id, resultado, gh, ga)
            resolvidos += n

            # Contar acertos (baseado no mercado previsto)
            mercado = pred["mercado"]
            total_gols = gh + ga

            # Verificar acerto — todos os mercados possíveis
            acertou = False
            if mercado == "h2h_home" and resultado == "home":
                acertou = True
            elif mercado == "h2h_draw" and resultado == "draw":
                acertou = True
            elif mercado == "h2h_away" and resultado == "away":
                acertou = True
            elif mercado == "over15" and total_gols > 1:
                acertou = True
            elif mercado == "under15" and total_gols < 2:
                acertou = True
            elif mercado == "over25" and total_gols > 2:
                acertou = True
            elif mercado == "under25" and total_gols < 3:
                acertou = True
            elif mercado == "over35" and total_gols > 3:
                acertou = True
            elif mercado == "under35" and total_gols < 4:
                acertou = True
            elif mercado == "btts_yes" and gh > 0 and ga > 0:
                acertou = True
            elif mercado == "btts_no" and (gh == 0 or ga == 0):
                acertou = True
            elif mercado == "ht_home" and resultado == "home":
                acertou = True
            elif mercado == "ht_draw" and resultado == "draw":
                acertou = True
            elif mercado == "ht_away" and resultado == "away":
                acertou = True

            if acertou:
                acertos += 1
            else:
                erros += 1

        return {
            "total_pendentes": len(pendentes),
            "resolvidos": resolvidos,
            "acertos": acertos,
            "erros": erros,
            "accuracy": round(acertos / max(resolvidos, 1) * 100, 1),
        }

    def relatorio_diario(self) -> str:
        """
        Gera relatório detalhado de performance para o Telegram.

        Filtra métricas pela versão do modelo ativo, ignorando
        previsões feitas por modelos anteriores (evita ruído).

        Inclui: métricas gerais, breakdown por mercado, por liga,
        evolução recente vs histórico, e status do modelo.
        """
        # Obter versão do modelo ativo para contexto correto
        treino = self.db.ultimo_treino()
        versao = treino["modelo_versao"] if treino else None

        total = self.db.metricas_modelo(modelo_versao=versao)
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━",
            "📊 <b>RELATÓRIO DE PERFORMANCE</b>",
            f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
        ]

        if total["total"] == 0:
            lines.append("📭 Nenhuma previsão resolvida ainda.")
            resumo = self.db.resumo()
            lines.extend([
                "",
                f"📦 Banco: {resumo['fixtures']:,} fixtures | {resumo['fixtures_com_stats']:,} com stats",
            ])
            return "\n".join(lines)

        # ─── Resumo geral ───
        emoji_roi = "🟢" if total['roi'] > 0 else "🔴" if total['roi'] < -5 else "🟡"
        lines.extend([
            f"<b>📈 GERAL</b>",
            f"  Apostas: {total['total']} | Acertos: {total['acertos']}",
            f"  Accuracy: <b>{total['accuracy']}%</b>",
            f"  {emoji_roi} ROI: <b>{total['roi']:+.1f}%</b>",
            f"  Lucro: <b>{total['lucro_total']:+.2f}u</b>",
            "",
        ])

        # ─── Breakdown por mercado × liga ───
        # Nomes legíveis para cada mercado
        nomes_mercado = {
            "h2h_home": "Casa",
            "h2h_draw": "Empate",
            "h2h_away": "Fora",
            "over25": "Over 2.5",
            "under25": "Under 2.5",
            "btts_yes": "BTTS Sim",
            "btts_no": "BTTS Não",
        }

        lines.append("<b>🎯 POR MERCADO</b>")

        # Métricas globais por mercado (resumo)
        melhor_mercado = None
        melhor_roi = -999
        for key, label in nomes_mercado.items():
            m = self.db.metricas_modelo(mercado=key, modelo_versao=versao)
            if m["total"] > 0:
                emoji = "✅" if m["roi"] > 0 else "❌"
                lines.append(f"  {emoji} {label}: {m['accuracy']}% ({m['acertos']}/{m['total']}) ROI: {m['roi']:+.1f}%")
                if m["roi"] > melhor_roi:
                    melhor_roi = m["roi"]
                    melhor_mercado = label

        if melhor_mercado:
            lines.append(f"  🏆 Melhor: <b>{melhor_mercado}</b> ({melhor_roi:+.1f}%)")
        lines.append("")

        # Detalhamento mercado × liga (mín. 3 previsões)
        dados_ml = self.db.metricas_por_mercado_liga(min_amostras=3,
                                                      modelo_versao=versao)
        if dados_ml:
            lines.append("<b>🏟 MERCADO × LIGA</b>")
            mercado_atual = None
            for d in dados_ml:
                label = nomes_mercado.get(d["mercado"], d["mercado"])
                # Cabeçalho do mercado quando muda
                if d["mercado"] != mercado_atual:
                    mercado_atual = d["mercado"]
                    lines.append(f"  <b>{label}</b>")
                # Emoji de performance
                emoji = "🟢" if d["accuracy"] >= 55 else "🟡" if d["accuracy"] >= 45 else "🔴"
                liga = d["league_name"] or f"Liga {d['league_id']}"
                lines.append(
                    f"    {emoji} {liga}: {d['accuracy']}% "
                    f"({d['acertos']}/{d['total']}) ROI: {d['roi']:+.1f}%"
                )
            lines.append("")

        # ─── Evolução recente vs histórico ───
        janela = self._metricas_janela(
            (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            modelo_versao=versao,
        )
        if janela["total"] > 0:
            tendencia = "📈" if janela["roi"] > total["roi"] else "📉" if janela["roi"] < total["roi"] else "➡️"
            lines.extend([
                f"<b>{tendencia} ÚLTIMOS 7 DIAS</b>",
                f"  Apostas: {janela['total']} | Acertos: {janela['acertos']}",
                f"  Accuracy: {janela['accuracy']:.1f}% | ROI: {janela['roi']:+.1f}%",
                "",
            ])

        # ─── Status do modelo ───
        treino = self.db.ultimo_treino()
        if treino:
            # Data do treino em formato BR (dd/mm/yyyy)
            try:
                _d = datetime.strptime(treino['date'][:10], '%Y-%m-%d')
                data_treino_br = _d.strftime('%d/%m/%Y')
            except Exception:
                data_treino_br = treino['date'][:10]
            lines.extend([
                f"<b>🤖 MODELO</b>",
                f"  Versão: {treino['modelo_versao']}",
                f"  Treino: {data_treino_br}",
                f"  Acc train/test: {(treino['accuracy_train'] or 0)*100:.1f}% / {(treino['accuracy_test'] or 0)*100:.1f}%",
            ])

        # ─── Sequência atual ───
        seq = self._sequencia_derrotas()
        if seq >= 3:
            lines.append(f"  🔴 Sequência: {seq} erros consecutivos")
        elif seq == 0:
            # Contar acertos seguidos
            conn = self.db._conn()
            rows = conn.execute("""
                SELECT acertou FROM predictions
                WHERE acertou IS NOT NULL
                ORDER BY date DESC, created_at DESC LIMIT 10
            """).fetchall()
            conn.close()
            seq_ok = 0
            for r in rows:
                if r["acertou"] == 1:
                    seq_ok += 1
                else:
                    break
            if seq_ok >= 3:
                lines.append(f"  🟢 Sequência: {seq_ok} acertos consecutivos!")

        # ─── Banco ───
        resumo = self.db.resumo()
        lines.extend([
            "",
            f"<b>📦 BANCO</b>",
            f"  {resumo['fixtures']:,} fixtures | {resumo['fixtures_com_stats']:,} stats | {resumo['predictions']:,} previsões",
        ])

        return "\n".join(lines)

    def verificar_degradacao(self) -> dict:
        """
        Verifica se o modelo está degradando numa janela recente.

        Analisa previsões dos últimos DEGRADATION_WINDOW_DAYS dias e compara
        accuracy com o limiar mínimo. Também verifica ROI para auto-pause.

        Retorna:
          {
            "degradado": bool,
            "pausado": bool,
            "alertas": list[str],
            "metricas_janela": dict,
            "metricas_geral": dict,
          }
        """
        alertas = []
        degradado = False
        pausado = False

        # Métricas gerais (apenas do modelo ativo)
        treino = self.db.ultimo_treino()
        versao = treino["modelo_versao"] if treino else None
        metricas_geral = self.db.metricas_modelo(modelo_versao=versao)

        # Métricas da janela recente (também filtradas pela versão ativa)
        data_inicio = (datetime.now() - timedelta(days=DEGRADATION_WINDOW_DAYS)).strftime("%Y-%m-%d")
        metricas_janela = self._metricas_janela(data_inicio, modelo_versao=versao)

        # Verificação 1: Accuracy da janela recente
        if metricas_janela["total"] >= 5:  # Precisa de pelo menos 5 previsões na janela
            if metricas_janela["accuracy"] < DEGRADATION_ACC_MIN * 100:
                degradado = True
                alertas.append(
                    f"⚠️ Accuracy dos últimos {DEGRADATION_WINDOW_DAYS} dias "
                    f"({metricas_janela['accuracy']:.1f}%) abaixo do mínimo "
                    f"({DEGRADATION_ACC_MIN*100:.0f}%). Considere retreinar."
                )

        # Verificação 2: ROI para auto-pause
        if metricas_geral["total"] >= ROI_PAUSE_MIN_BETS:
            if metricas_geral["roi"] < ROI_PAUSE_THRESHOLD:
                pausado = True
                alertas.append(
                    f"🛑 ROI acumulado ({metricas_geral['roi']:+.1f}%) abaixo de "
                    f"{ROI_PAUSE_THRESHOLD}%. Scanner será PAUSADO até retreino."
                )

        # Verificação 3: Sequência de derrotas (5+ seguidas = alerta)
        seq_derrotas = self._sequencia_derrotas()
        if seq_derrotas >= 5:
            alertas.append(
                f"🔴 {seq_derrotas} previsões erradas consecutivas. "
                f"Modelo pode estar descalibrado."
            )
            degradado = True

        # Se tudo bem
        if not alertas:
            alertas.append("✅ Modelo operando dentro dos parâmetros normais.")

        return {
            "degradado": degradado,
            "pausado": pausado,
            "alertas": alertas,
            "metricas_janela": metricas_janela,
            "metricas_geral": metricas_geral,
        }

    def relatorio_saude(self) -> str:
        """
        Gera relatório de saúde do modelo para envio no Telegram.
        Chamado pelo scheduler no relatório noturno.
        """
        check = self.verificar_degradacao()

        lines = [
            "🏥 *Saúde do Modelo*",
            f"📅 {datetime.now().strftime('%d/%m/%Y')}",
            "",
        ]

        for alerta in check["alertas"]:
            lines.append(alerta)

        mj = check["metricas_janela"]
        mg = check["metricas_geral"]

        if mj["total"] > 0:
            lines.extend([
                "",
                f"*Últimos {DEGRADATION_WINDOW_DAYS} dias:*",
                f"  Previsões: {mj['total']} | Acertos: {mj['acertos']}",
                f"  Accuracy: {mj['accuracy']:.1f}% | ROI: {mj['roi']:+.1f}%",
            ])

        if mg["total"] > 0:
            lines.extend([
                "",
                f"*Acumulado total:*",
                f"  Previsões: {mg['total']} | Acertos: {mg['acertos']}",
                f"  Accuracy: {mg['accuracy']}% | ROI: {mg['roi']:+.1f}%",
                f"  Lucro: {mg['lucro_total']:+.2f}u",
            ])

        return "\n".join(lines)

    def _metricas_janela(self, data_inicio: str,
                          modelo_versao: str = None) -> dict:
        """
        Calcula métricas apenas para previsões a partir de uma data.

        Parâmetros:
          data_inicio: data ISO (YYYY-MM-DD) para filtro temporal
          modelo_versao: se informado, filtra apenas previsões desta versão
        """
        sql = "SELECT * FROM predictions WHERE acertou IS NOT NULL AND date >= ?"
        params = [data_inicio]
        if modelo_versao:
            sql += " AND modelo_versao = ?"
            params.append(modelo_versao)

        conn = self.db._conn()
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        if not rows:
            return {"total": 0, "acertos": 0, "accuracy": 0, "roi": 0}

        total = len(rows)
        acertos = sum(1 for r in rows if r["acertou"] == 1)
        lucro = sum(r["lucro"] or 0 for r in rows)

        return {
            "total": total,
            "acertos": acertos,
            "accuracy": round(acertos / total * 100, 1),
            "roi": round(lucro / total * 100, 1),
        }

    def _sequencia_derrotas(self) -> int:
        """Conta quantas previsões erradas consecutivas (da mais recente para trás)."""
        conn = self.db._conn()
        rows = conn.execute("""
            SELECT acertou FROM predictions
            WHERE acertou IS NOT NULL
            ORDER BY date DESC, created_at DESC
            LIMIT 20
        """).fetchall()
        conn.close()

        seq = 0
        for r in rows:
            if r["acertou"] == 0:
                seq += 1
            else:
                break
        return seq

    @staticmethod
    def _confidence_from_prediction(row: dict) -> float | None:
        """Retorna a confiança correta da tip salva, sem misturar mercados."""
        prob = row.get("prob_modelo")
        if prob is not None:
            return prob

        fallback_map = {
            "h2h_home": "prob_home",
            "h2h_draw": "prob_draw",
            "h2h_away": "prob_away",
            "over25": "prob_over25",
            "btts_yes": "prob_btts",
            "btts_no": "prob_btts",
        }
        key = fallback_map.get(row.get("mercado"))
        if not key:
            return None
        return row.get(key)

    def relatorio_resultado_dia(self, data: str = None) -> str:
        """
        Gera relatório detalhado dos resultados do dia.

        Inclui: cada jogo com resultado, análise de acerto/erro,
        breakdown por mercado, lucro do dia e comparação com acumulado.
        """
        if data is None:
            data = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        conn = self.db._conn()
        raw_rows = conn.execute("""
            SELECT * FROM predictions
            WHERE date LIKE ? AND acertou IS NOT NULL
            ORDER BY ev_percent DESC
        """, (f"{data}%",)).fetchall()
        conn.close()

        # Converter sqlite3.Row para dict (evita erro com .get())
        rows = [dict(r) for r in raw_rows]

        # Formatar data em DD/MM para exibição
        try:
            _dt = datetime.strptime(data, "%Y-%m-%d")
            data_br = _dt.strftime("%d/%m")
        except ValueError:
            data_br = data

        if not rows:
            return f"📋 Nenhum resultado para {data_br}"

        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📋 <b>RESULTADOS — {data_br}</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
        ]

        acertos = 0
        lucro = 0
        por_mercado = {}

        for r in rows:
            acertou = r["acertou"] == 1
            emoji = "✅" if acertou else "❌"
            lucro_item = r["lucro"] or 0
            mercado = r["mercado"] or "?"
            odd = r["odd_usada"] or 0
            ev = r["ev_percent"] or 0

            # Nome do mercado legível — inclui todos os mercados possíveis
            nomes_mercado = {
                "h2h_home": "Casa", "h2h_draw": "Empate", "h2h_away": "Fora",
                "over15": "Over 1.5", "under15": "Under 1.5",
                "over25": "Over 2.5", "under25": "Under 2.5",
                "over35": "Over 3.5", "under35": "Under 3.5",
                "btts_yes": "BTTS Sim", "btts_no": "BTTS Não",
                "ht_home": "1T Casa", "ht_draw": "1T Empate", "ht_away": "1T Fora",
            }
            mercado_label = nomes_mercado.get(mercado, mercado)

            # Placar
            gh = r.get("gols_home")
            ga = r.get("gols_away")
            placar = f"{gh}-{ga}" if gh is not None and ga is not None else "?"

            acertou_txt = "✅ Acertou!" if acertou else "❌ Errou"
            prob = self._confidence_from_prediction(r)
            if prob is None:
                conf_txt = "n/d"
            else:
                conf_pct = prob * 100 if prob <= 1 else prob
                conf_txt = f"{conf_pct:.0f}%"

            lines.append(
                f"{emoji} <b>{r['home_name']} vs {r['away_name']}</b> ({placar})\n"
                f"   Aposta: {mercado_label} | Confiança: {conf_txt}\n"
                f"   {acertou_txt}"
            )

            if acertou:
                acertos += 1
            lucro += lucro_item

            # Agrupar por mercado
            if mercado not in por_mercado:
                por_mercado[mercado] = {"total": 0, "acertos": 0, "lucro": 0}
            por_mercado[mercado]["total"] += 1
            por_mercado[mercado]["acertos"] += int(acertou)
            por_mercado[mercado]["lucro"] += lucro_item

        # ─── Resumo do dia ───
        total = len(rows)
        pct = acertos / total * 100 if total > 0 else 0
        emoji_dia = "🟢" if pct >= 55 else "🔴" if pct < 40 else "🟡"
        lines.extend([
            "",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"{emoji_dia} <b>RESUMO DO DIA</b>",
            f"  Tips: {total} | Acertos: {acertos}/{total} ({pct:.0f}%)",
        ])

        # Breakdown por mercado (se mais de 1 tipo)
        if len(por_mercado) > 1:
            lines.append("")
            lines.append("<b>Por mercado:</b>")
            for m, dados in sorted(por_mercado.items()):
                label = nomes_mercado.get(m, m)
                pct_m = dados['acertos'] / dados['total'] * 100 if dados['total'] > 0 else 0
                lines.append(
                    f"  {label}: {dados['acertos']}/{dados['total']} ({pct_m:.0f}%)"
                )

        # Comparar com acumulado geral (modelo ativo)
        treino_info = self.db.ultimo_treino()
        versao_ativa = treino_info["modelo_versao"] if treino_info else None
        geral = self.db.metricas_modelo(modelo_versao=versao_ativa)
        if geral["total"] > 0:
            lines.extend([
                "",
                f"<b>📈 Acumulado:</b> {geral['accuracy']}% accuracy | {geral['total']} tips total",
            ])

        return "\n".join(lines)
