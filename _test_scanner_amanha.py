"""Teste: rodar scanner para amanhã e ver output formatado."""
import sys, os
sys.path.insert(0, '/opt/futebot')
os.environ.setdefault('API_FOOTBALL_KEY', '7d40d96b3852438ee6fd1d4896bc54b9')

from pipeline.scanner import Scanner

s = Scanner()
resultado = s.executar(dias_adiante=1)
msgs = s.formatar_relatorio(resultado)
for m in msgs:
    print(m)
    print()
