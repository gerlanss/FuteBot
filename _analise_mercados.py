"""Análise de performance por mercado — consulta direto no banco do VPS."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

query = r"""
cd /opt/futebot && /opt/futebot/venv/bin/python3 -c "
import sqlite3, json

conn = sqlite3.connect('data/futebot.db')
conn.row_factory = sqlite3.Row

# 1. Performance por mercado (resolvidas)
print('=== PERFORMANCE POR MERCADO (RESOLVIDAS) ===')
print(f'{\"Mercado\":<20} {\"Total\":>5} {\"Acertos\":>7} {\"Erros\":>5} {\"Acc%\":>6} {\"Lucro\":>8} {\"ROI%\":>7} {\"OddMed\":>7}')
print('-' * 80)

rows = conn.execute('''
    SELECT 
        mercado,
        COUNT(*) as total,
        COALESCE(SUM(CASE WHEN acertou = 1 THEN 1 ELSE 0 END), 0) as acertos,
        COALESCE(SUM(CASE WHEN acertou = 0 THEN 1 ELSE 0 END), 0) as erros,
        ROUND(AVG(CASE WHEN acertou IS NOT NULL THEN acertou END) * 100, 1) as accuracy,
        COALESCE(ROUND(SUM(CASE WHEN lucro IS NOT NULL THEN lucro ELSE 0 END), 2), 0) as lucro_total,
        COALESCE(ROUND(AVG(odd_usada), 2), 0) as odd_media
    FROM predictions
    WHERE acertou IS NOT NULL
    GROUP BY mercado
    ORDER BY total DESC
''').fetchall()

total_geral = 0
acertos_geral = 0
lucro_geral = 0.0
for r in rows:
    t = r['total'] or 0
    a = r['acertos'] or 0
    e2 = r['erros'] or 0
    acc = r['accuracy'] or 0
    luc = r['lucro_total'] or 0
    odd = r['odd_media'] or 0
    roi = (luc / t * 100) if t > 0 else 0
    total_geral += t
    acertos_geral += a
    lucro_geral += luc
    print(f'{r[\"mercado\"]:<20} {t:>5} {a:>7} {e2:>5} {acc:>5.1f}% {luc:>+7.2f} {roi:>+6.1f}% {odd:>6.2f}')

roi_geral = (lucro_geral / total_geral * 100) if total_geral > 0 else 0
acc_geral = (acertos_geral / total_geral * 100) if total_geral > 0 else 0
print('-' * 80)
print(f'{\"TOTAL\":<20} {total_geral:>5} {acertos_geral:>7} {total_geral - acertos_geral:>5} {acc_geral:>5.1f}% {lucro_geral:>+7.2f} {roi_geral:>+6.1f}%')

# 2. Tips pendentes (não resolvidas)
print()
print('=== TIPS PENDENTES (NÃO RESOLVIDAS) ===')
rows2 = conn.execute('''
    SELECT mercado, COUNT(*) as n
    FROM predictions
    WHERE acertou IS NULL
    GROUP BY mercado
    ORDER BY n DESC
''').fetchall()
for r in rows2:
    print(f'  {r[\"mercado\"]:<20} {r[\"n\"]:>3} pendentes')

# 3. Distribuição por modelo_versao
print()
print('=== DISTRIBUIÇÃO POR VERSÃO DO MODELO ===')
rows3 = conn.execute('''
    SELECT modelo_versao, COUNT(*) as n,
        SUM(CASE WHEN acertou=1 THEN 1 ELSE 0 END) as ok,
        SUM(CASE WHEN acertou=0 THEN 1 ELSE 0 END) as nok,
        SUM(CASE WHEN acertou IS NULL THEN 1 ELSE 0 END) as pend
    FROM predictions
    GROUP BY modelo_versao
    ORDER BY modelo_versao DESC
''').fetchall()
for r in rows3:
    acc = (r['ok'] / (r['ok'] + r['nok']) * 100) if (r['ok'] + r['nok']) > 0 else 0
    print(f'  {r[\"modelo_versao\"]:<25} Total:{r[\"n\"]:>4}  Acc:{acc:>5.1f}%  OK:{r[\"ok\"]:>3}  NOK:{r[\"nok\"]:>3}  Pend:{r[\"pend\"]:>3}')

# 4. Strategies ativas por mercado
print()
print('=== STRATEGIES ATIVAS POR MERCADO ===')
rows4 = conn.execute('''
    SELECT mercado, COUNT(*) as n, 
        ROUND(AVG(accuracy), 1) as acc_media,
        ROUND(MIN(conf_min), 2) as conf_min,
        ROUND(MAX(conf_max), 2) as conf_max
    FROM strategies
    WHERE ativo = 1
    GROUP BY mercado
    ORDER BY n DESC
''').fetchall()
for r in rows4:
    print(f'  {r[\"mercado\"]:<20} {r[\"n\"]:>3} estratégias  Acc média: {r[\"acc_media\"]:>5.1f}%  Conf: [{r[\"conf_min\"]:.2f}-{r[\"conf_max\"]:.2f}]')

conn.close()
"
"""

_, o, e = ssh.exec_command(query)
print(o.read().decode())
err = e.read().decode().strip()
if err:
    print("STDERR:", err)

ssh.close()
