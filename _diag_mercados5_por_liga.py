"""Diagnóstico v5: probabilidades POR LIGA × MERCADO (não global)."""
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

# Pegar todas as ligas ativas
from config import LEAGUES
liga_ids = {v['id']: k for k, v in LEAGUES.items()}
liga_nomes = {}
for key, val in LEAGUES.items():
    liga_nomes[val['id']] = val.get('nome', key)

# Fixtures FT recentes (ultimos 30 dias) — max 200 por liga
rows = conn.execute('''
    SELECT fixture_id, home_name, away_name, home_id, away_id,
           league_id, season, date
    FROM fixtures
    WHERE date >= '2026-02-01' AND date <= '2026-03-04'
    AND status = 'FT'
    AND league_id IN ({})
    ORDER BY league_id, date DESC
'''.format(','.join(str(lid) for lid in liga_ids.keys()))).fetchall()
conn.close()

print(f'Total fixtures FT: {len(rows)}')

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

# Agrupar por liga
from collections import defaultdict
por_liga = defaultdict(list)
for row in rows:
    por_liga[row['league_id']].append(row)

# Rodar predictor por liga
# Estrutura:  liga_mercado_probs[league_id][mercado] = [prob, prob, ...]
liga_mercado_probs = {}
liga_jogos_ok = {}

for league_id in sorted(por_liga.keys()):
    fixtures = por_liga[league_id]
    mercado_probs = defaultdict(list)
    jogos_ok = 0

    for row in fixtures:
        fix_dict = {
            'fixture_id': row['fixture_id'],
            'home_id': row['home_id'],
            'away_id': row['away_id'],
            'home_name': row['home_name'],
            'away_name': row['away_name'],
            'league_id': row['league_id'],
            'season': row['season'] if row['season'] else 2025,
            'round': '',
            'date': row['date'],
        }
        try:
            resultado = pred.prever_jogo(fix_dict)
            if not resultado:
                continue
            jogos_ok += 1
            for prob_key, mercado_name in MERCADOS_CHECK:
                v = resultado.get(prob_key, None)
                if v is not None:
                    mercado_probs[mercado_name].append(v)
        except:
            pass

    liga_mercado_probs[league_id] = mercado_probs
    liga_jogos_ok[league_id] = jogos_ok

# Strategy gate atual
strats = db.strategies_ativas()
strat_set = set()
for s in strats:
    strat_set.add((s['mercado'], s['league_id']))

# Imprimir por liga
for league_id in sorted(liga_mercado_probs.keys()):
    nome = liga_nomes.get(league_id, f'Liga {league_id}')
    jogos = liga_jogos_ok.get(league_id, 0)
    total_fix = len(por_liga[league_id])
    print(f'\n{\"=\" * 80}')
    print(f'LIGA {league_id}: {nome} ({jogos}/{total_fix} jogos com predicao)')
    print(f'{\"=\" * 80}')
    print(f'{\"Mercado\":<18} {\"N\":>3} {\"Min\":>6} {\"Avg\":>6} {\"Max\":>6} {\">=60%\":>5} {\"Gate\":>5} Status')
    print('-' * 75)

    mercado_probs = liga_mercado_probs[league_id]
    for prob_key, mercado_name in MERCADOS_CHECK:
        vals = mercado_probs.get(mercado_name, [])
        tem_gate = 'SIM' if (mercado_name, league_id) in strat_set else '--'
        if not vals:
            print(f'{mercado_name:<18} {0:>3}   ---    ---    ---   ---  {tem_gate:>5} sem probs')
            continue
        mn = min(vals)
        mx = max(vals)
        avg = sum(vals)/len(vals)
        ge60 = sum(1 for v in vals if v >= 0.60)

        if ge60 == 0:
            status = f'BLOQ(max {mx:.0%})'
        elif tem_gate == '--':
            status = f'SEM GATE! {ge60} tips perdidas'
        else:
            status = f'OK {ge60} tips'

        print(f'{mercado_name:<18} {len(vals):>3} {mn:>6.1%} {avg:>6.1%} {mx:>6.1%} {ge60:>5}  {tem_gate:>5} {status}')

# Resumo: mercados com tips perdidas (sem gate mas com probs >= 60%)
print(f'\n{\"=\" * 80}')
print('RESUMO: TIPS PERDIDAS (mercado gera prob >=60% mas SEM strategy gate)')
print(f'{\"=\" * 80}')
print(f'{\"Liga\":<30} {\"Mercado\":<18} {\"Jogos\":>5} {\">=60%\":>5} {\"Avg\":>6} {\"Max\":>6}')
print('-' * 80)

total_perdidas = 0
for league_id in sorted(liga_mercado_probs.keys()):
    nome = liga_nomes.get(league_id, f'Liga {league_id}')
    mercado_probs = liga_mercado_probs[league_id]
    for prob_key, mercado_name in MERCADOS_CHECK:
        vals = mercado_probs.get(mercado_name, [])
        if not vals:
            continue
        ge60 = sum(1 for v in vals if v >= 0.60)
        if ge60 > 0 and (mercado_name, league_id) not in strat_set:
            avg = sum(vals)/len(vals)
            mx = max(vals)
            total_perdidas += ge60
            print(f'{nome:<30} {mercado_name:<18} {len(vals):>5} {ge60:>5} {avg:>6.1%} {mx:>6.1%}')

print(f'\nTotal tips perdidas por falta de strategy gate: {total_perdidas}')
"
"""

_, stdout, stderr = ssh.exec_command(query, timeout=180)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    print("STDERR:", err[-2000:])
ssh.close()
