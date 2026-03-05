"""Diagnóstico: por que mercados não geram tips? Analisa o predictor ao vivo."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

query = r"""
cd /opt/futebot && /opt/futebot/venv/bin/python3 -c "
import sys, os, json
sys.path.insert(0, '.')
os.environ.setdefault('API_FOOTBALL_KEY', '7d40d96b3852438ee6fd1d4896bc54b9')

from data.database import Database
from models.predictor import Predictor
from pipeline.scanner import PROB_MIN, CONF_MIN_ABSOLUTA

db = Database()
pred = Predictor(db)

# Pegar fixtures de amanhã ou hoje para testar
import sqlite3
conn = db._conn()
rows = conn.execute('''
    SELECT fixture_id, home_name, away_name, league_id, date
    FROM fixtures
    WHERE date >= date('now', '-2 days') AND date <= date('now', '+1 day')
    AND status IN ('NS', 'FT', 'TBD')
    ORDER BY date DESC
    LIMIT 20
''').fetchall()
conn.close()

print(f'=== PROBABILIDADES BRUTAS DO PREDICTOR ({len(rows)} fixtures) ===')
print(f'Threshold PROB_MIN = {PROB_MIN}')
print(f'Threshold CONF_MIN = {CONF_MIN_ABSOLUTA}')
print()

# Agrupar todas as probs por mercado para ver distribuição
from collections import defaultdict
mercado_probs = defaultdict(list)

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

for row in rows:
    fix = {'fixture': {'id': row['fixture_id']}, 'league': {'id': row['league_id']},
           'teams': {'home': {'name': row['home_name']}, 'away': {'name': row['away_name']}}}
    try:
        resultado = pred.prever_jogo(fix)
        if not resultado:
            continue
        for prob_key, mercado_name in MERCADOS_CHECK:
            v = resultado.get(prob_key, None)
            if v is not None:
                mercado_probs[mercado_name].append(v)
    except Exception as e:
        pass

print(f'Jogos analisados com sucesso: {sum(1 for v in mercado_probs.values() if v)}')
print()
print(f'{\"Mercado\":<18} {\"N\":>3} {\"Min\":>6} {\"Avg\":>6} {\"Max\":>6} {\">=60%\":>5} {\">=55%\":>5} {\">=50%\":>5} {\"Status\"}')
print('-' * 85)

for prob_key, mercado_name in MERCADOS_CHECK:
    vals = mercado_probs.get(mercado_name, [])
    if not vals:
        print(f'{mercado_name:<18} {0:>3}  -- sem dados --')
        continue
    mn = min(vals)
    mx = max(vals)
    avg = sum(vals)/len(vals)
    ge60 = sum(1 for v in vals if v >= 0.60)
    ge55 = sum(1 for v in vals if v >= 0.55)
    ge50 = sum(1 for v in vals if v >= 0.50)
    status = 'PASSA' if ge60 > 0 else 'BLOQUEADO (max < 60%)'
    print(f'{mercado_name:<18} {len(vals):>3} {mn:>5.1%} {avg:>5.1%} {mx:>5.1%} {ge60:>5} {ge55:>5} {ge50:>5}  {status}')

# Strategy Gate: quantas estratégias ativas por mercado bloqueado?
print()
print('=== STRATEGY GATE: Mercados sem estratégias ativas ===')
conn = db._conn()
rows2 = conn.execute('SELECT DISTINCT mercado FROM strategies WHERE ativo=1').fetchall()
conn.close()
ativos = {r['mercado'] for r in rows2}
todos = {m for _, m in MERCADOS_CHECK}
sem_gate = todos - ativos
if sem_gate:
    for m in sorted(sem_gate):
        print(f'  ❌ {m} — sem strategy gate ativo (bloqueado na etapa 4)')
else:
    print('  Todos os mercados têm pelo menos 1 strategy ativa')
"
"""

_, o, e = ssh.exec_command(query, timeout=60)
print(o.read().decode())
err = e.read().decode().strip()
if err:
    # Filtrar warnings de xgboost
    lines = [l for l in err.split('\n') if 'WARNING' not in l.upper() and l.strip()]
    if lines:
        print("STDERR:", '\n'.join(lines[-10:]))

ssh.close()
