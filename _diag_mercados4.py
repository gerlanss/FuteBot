"""Diagnóstico v4: fixture dict FLAT (como o scanner monta)."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

query = r"""
cd /opt/futebot && /opt/futebot/venv/bin/python3 -c "
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('API_FOOTBALL_KEY', '7d40d96b3852438ee6fd1d4896bc54b9')

from data.database import Database
from models.predictor import Predictor
from collections import defaultdict

db = Database()
pred = Predictor(db)

conn = db._conn()
rows = conn.execute('''
    SELECT fixture_id, home_name, away_name, home_id, away_id,
           league_id, season, date, status
    FROM fixtures
    WHERE date >= '2026-02-25' AND date <= '2026-03-04'
    AND status = 'FT'
    ORDER BY date DESC
    LIMIT 30
''').fetchall()
conn.close()

print(f'Fixtures FT encontrados: {len(rows)}')

MERCADOS_CHECK = [
    ('prob_home', 'h2h_home'), ('prob_draw', 'h2h_draw'), ('prob_away', 'h2h_away'),
    ('prob_over15', 'over15'), ('prob_under15', 'under15'),
    ('prob_over25', 'over25'), ('prob_under25', 'under25'),
    ('prob_over35', 'over35'), ('prob_under35', 'under35'),
    ('prob_btts_yes', 'btts_yes'), ('prob_btts_no', 'btts_no'),
    ('prob_ht_home', 'ht_home'), ('prob_ht_draw', 'ht_draw'), ('prob_ht_away', 'ht_away'),
    ('prob_over05_ht', 'over05_ht'), ('prob_under05_ht', 'under05_ht'),
    ('prob_over15_ht', 'over15_ht'), ('prob_under15_ht', 'under15_ht'),
    ('prob_over05_2t', 'over05_2t'), ('prob_under05_2t', 'under05_2t'),
    ('prob_over15_2t', 'over15_2t'), ('prob_under15_2t', 'under15_2t'),
    ('prob_corners_over_85', 'corners_o85'), ('prob_corners_under_85', 'corners_u85'),
    ('prob_corners_over_95', 'corners_o95'), ('prob_corners_under_95', 'corners_u95'),
    ('prob_corners_over_105', 'corners_o105'), ('prob_corners_under_105', 'corners_u105'),
]

mercado_probs = defaultdict(list)
jogos_ok = 0
erros = 0

for row in rows:
    fix_dict = {
        'fixture_id': row['fixture_id'],
        'home_id': row['home_id'],
        'away_id': row['away_id'],
        'home_name': row['home_name'],
        'away_name': row['away_name'],
        'league_id': row['league_id'],
        'season': row['season'],
        'round': '',
        'date': row['date'],
    }
    try:
        resultado = pred.prever_jogo(fix_dict)
        if not resultado:
            erros += 1
            if erros <= 3:
                print(f'SEM DADOS: {row[\"home_name\"]} vs {row[\"away_name\"]} (liga {row[\"league_id\"]})')
            continue
        jogos_ok += 1
        for prob_key, mercado_name in MERCADOS_CHECK:
            v = resultado.get(prob_key, None)
            if v is not None:
                mercado_probs[mercado_name].append(v)
        if jogos_ok == 1:
            print(f'Exemplo: {row[\"home_name\"]} vs {row[\"away_name\"]} (liga {row[\"league_id\"]})')
            for k, v in sorted(resultado.items()):
                if k.startswith('prob_'):
                    print(f'  {k}: {v}')
            print()
    except Exception as e:
        erros += 1
        if erros <= 5:
            print(f'ERRO: {row[\"home_name\"]} vs {row[\"away_name\"]} -> {type(e).__name__}: {e}')

print(f'Jogos com previsao: {jogos_ok}/{len(rows)} (erros/sem dados: {erros})')
print()
print(f'{\"Mercado\":<18} {\"N\":>3} {\"Min\":>6} {\"Avg\":>6} {\"Max\":>6} {\">=60%\":>5} {\">=55%\":>5} {\">=50%\":>5} Barreira')
print('-' * 90)

for prob_key, mercado_name in MERCADOS_CHECK:
    vals = mercado_probs.get(mercado_name, [])
    if not vals:
        print(f'{mercado_name:<18} {0:>3}  -- MODELO NAO GERA PROBS --')
        continue
    mn = min(vals)
    mx = max(vals)
    avg = sum(vals)/len(vals)
    ge60 = sum(1 for v in vals if v >= 0.60)
    ge55 = sum(1 for v in vals if v >= 0.55)
    ge50 = sum(1 for v in vals if v >= 0.50)
    if ge60 == 0:
        bar = f'BLOQ: max {mx:.0%} < 60%'
    elif ge60 < len(vals) * 0.1:
        bar = f'RARO: so {ge60}/{len(vals)} passam'
    else:
        bar = f'OK: {ge60}/{len(vals)} passam 60%'
    print(f'{mercado_name:<18} {len(vals):>3} {mn:>6.1%} {avg:>6.1%} {mx:>6.1%} {ge60:>5} {ge55:>5} {ge50:>5} {bar}')
"
"""

_, stdout, stderr = ssh.exec_command(query, timeout=120)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    print("STDERR:", err)
ssh.close()
