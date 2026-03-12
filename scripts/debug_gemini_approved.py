import json
from datetime import datetime
from pathlib import Path

from data.database import Database
from services.gemini_lookup import GeminiMarketLookup


def _load_context(row: dict) -> dict:
    raw = row.get("contexto_json")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _build_tip(row: dict, contexto: dict) -> dict:
    return {
        "fixture_id": row.get("fixture_id"),
        "league_id": row.get("league_id"),
        "home_name": row.get("home_name"),
        "away_name": row.get("away_name"),
        "mercado": row.get("mercado"),
        "descricao": row.get("descricao"),
        "date": contexto.get("date") or row.get("scan_date"),
    }


def main():
    db = Database()
    hoje = datetime.now().strftime("%Y-%m-%d")
    rows = [row for row in db.scan_audit_por_data(hoje) if row.get("approved_final")]
    lookup = GeminiMarketLookup()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data") / "gemini_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"approved_{stamp}.json"
    md_path = out_dir / f"approved_{stamp}.md"

    results = []
    md_lines = [f"# Gemini Debug Approved Tips - {hoje}", ""]

    for idx, row in enumerate(rows, start=1):
        contexto = _load_context(row)
        tip = _build_tip(row, contexto)
        debug = lookup.debug_lookup_market(tip)

        results.append({
            "tip": tip,
            "db_llm_decisao": row.get("llm_decisao"),
            "db_llm_motivo": row.get("llm_motivo"),
            "gemini_debug": debug,
        })

        md_lines.append(f"## Tip {idx}")
        md_lines.append(f"- Jogo: `{tip['home_name']} vs {tip['away_name']}`")
        md_lines.append(f"- Mercado: `{tip['mercado']}`")
        md_lines.append(f"- Descricao: `{tip['descricao']}`")
        md_lines.append(f"- DeepSeek: `{row.get('llm_decisao')}`")
        md_lines.append(f"- Motivo DeepSeek: {row.get('llm_motivo')}")
        md_lines.append("")
        md_lines.append(f"- Error geral: `{debug.get('error', '')}`")
        md_lines.append("")
        md_lines.append("### Merge Final")
        md_lines.append("```json")
        md_lines.append(json.dumps(debug.get("merged", {}), ensure_ascii=False, indent=2))
        md_lines.append("```")
        md_lines.append("")
        for stage in debug.get("stages", []):
            md_lines.append(f"### Etapa: {stage.get('stage')}")
            md_lines.append(f"- Parsed from: `{stage.get('parsed_from', 'n/a')}`")
            md_lines.append(f"- Error: `{stage.get('error', '')}`")
            md_lines.append("")
            md_lines.append("#### Prompt")
            md_lines.append("```text")
            md_lines.append(stage.get("prompt", ""))
            md_lines.append("```")
            md_lines.append("")
            md_lines.append("#### Queries")
            for query in stage.get("queries", []):
                md_lines.append(f"- {query}")
            if not stage.get("queries"):
                md_lines.append("- none")
            md_lines.append("")
            md_lines.append("#### Raw Text")
            md_lines.append("```text")
            md_lines.append(stage.get("raw_text", ""))
            md_lines.append("```")
            md_lines.append("")
            md_lines.append("#### Parsed")
            md_lines.append("```json")
            md_lines.append(json.dumps(stage.get("parsed", {}), ensure_ascii=False, indent=2))
            md_lines.append("```")
            md_lines.append("")
            md_lines.append("#### Sources")
            sources = stage.get("sources", [])
            if sources:
                for source in sources:
                    md_lines.append(f"- {source.get('title')}: {source.get('url')}")
            else:
                md_lines.append("- none")
            md_lines.append("")
            md_lines.append("#### Raw Payload")
            md_lines.append("```json")
            md_lines.append(json.dumps(stage.get("raw_payload", {}), ensure_ascii=False, indent=2))
            md_lines.append("```")
            md_lines.append("")

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({
        "date": hoje,
        "tips": len(rows),
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
