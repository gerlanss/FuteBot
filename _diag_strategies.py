"""Verifica mercados com/sem strategy gate."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

q = r"""
cd /opt/futebot && /opt/futebot/venv/bin/python3 -c "
import sys; sys.path.insert(0,'.')
from data.database import Database
db = Database()
conn = db._conn()

print('=== MERCADOS COM STRATEGY GATE (ativo=1) ===')
rows = conn.execute('''
    SELECT mercado, COUNT(*) as n, COUNT(DISTINCT league_id) as n_ligas,
           AVG(accuracy) as avg_acc, MIN(conf_min) as min_conf, MAX(conf_max) as max_conf
    FROM strategies WHERE ativo = 1
    GROUP BY mercado ORDER BY n DESC
''').fetchall()
for r in rows:
    print(f'  {r[\"mercado\"]:<20} slices={r[\"n\"]:>3}  ligas={r[\"n_ligas\"]:>2}  acc={r[\"avg_acc\"]:.1%}  conf=[{r[\"min_conf\"]:.2f}-{r[\"max_conf\"]:.2f}]')

print()
print('=== MERCADOS SEM NENHUMA STRATEGY (total) ===')
all_markets = [
    'h2h_home','h2h_draw','h2h_away',
    'over15','under15','over25','under25','over35','under35',
    'btts_yes','btts_no',
    'ht_home','ht_draw','ht_away',
    'over05_ht','under05_ht','over15_ht','under15_ht',
    'over05_2t','under05_2t','over15_2t','under15_2t',
    'corners_o85','corners_u85','corners_o95','corners_u95',
    'corners_o105','corners_u105',
]
existing = conn.execute('SELECT DISTINCT mercado FROM strategies').fetchall()
existing_set = {r['mercado'] for r in existing}
for m in all_markets:
    if m not in existing_set:
        print(f'  {m}  (ZERO entries)')

print()
print('=== TOTAL STRATEGIES ===')
total = conn.execute('SELECT COUNT(*) as n FROM strategies').fetchone()
ativas = conn.execute('SELECT COUNT(*) as n FROM strategies WHERE ativo=1').fetchone()
print(f'  Total: {total[\"n\"]}  Ativas: {ativas[\"n\"]}')
conn.close()
"
"""

_, stdout, stderr = ssh.exec_command(q, timeout=30)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    print("STDERR:", err)
ssh.close()
