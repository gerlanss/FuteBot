import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

# Enviar script
script = '''import sys
sys.path.insert(0, "/opt/futebot")

from pipeline.scanner import Scanner
from data.database import Database

db = Database()
scanner = Scanner(db)

print("Executando scanner...", flush=True)
resultado = scanner.executar(dias_adiante=0)

tips = resultado.get("tips", [])
print(f"Tips encontradas: {len(tips)}", flush=True)

if tips:
    msgs = scanner.formatar_relatorio(resultado)
    print(f"Mensagens: {len(msgs)}", flush=True)
    
    from services.telegram_bot import enviar_mensagem_direta
    import asyncio
    
    async def enviar():
        for msg in msgs:
            await enviar_mensagem_direta(msg)
            print(f"  Enviada: {msg[:60]}...", flush=True)
    
    asyncio.run(enviar())
    print("Tips enviadas!")
else:
    print("Nenhuma tip para hoje")
    jogos = resultado.get("jogos_analisados", 0)
    odds = resultado.get("jogos_com_odds", 0)
    print(f"Jogos analisados: {jogos}")
    print(f"Jogos com odds: {odds}")
'''

sftp = ssh.open_sftp()
with sftp.open('/opt/futebot/_run_scanner_now.py', 'w') as f:
    f.write(script)
sftp.close()
print("Script enviado")

# Rodar
_, o, _ = ssh.exec_command("cd /opt/futebot && nohup ./venv/bin/python -u _run_scanner_now.py > /tmp/scanner_manual.log 2>&1 &")
o.read()
time.sleep(2)

_, o, _ = ssh.exec_command("ps aux | grep scanner_now | grep -v grep")
ps = o.read().decode().strip()
print("Processo:", ps if ps else "Nao encontrado ou ja terminou")

ssh.close()
