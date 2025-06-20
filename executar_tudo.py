import subprocess
import os
import sys

print("\n🚀 Iniciando processo completo de inferência de comportamento + anomalia...")

try:
    # Executa o script responsável por mapear o comportamento padrão por conta
    subprocess.run([sys.executable, "inferencia/inferencia_comportamento.py"], check=True) 

    # Após mapear o comportamento, parte para a detecção de anomalia
    print("\n🚨 Etapa 2: Inferência de anomalias")
    subprocess.run([sys.executable, "inferencia/inferencia_anomalia.py"], check=True)

    # Mensagem final indicando sucesso na execução das etapas
    print("\n✅ Processo finalizado com sucesso!")
    print("\n📁 Resultados salvos na pasta: /resultados")

except subprocess.CalledProcessError as e:
    # Caso algum dos scripts falhe, exibe o erro ocorrido
    print("\n❌ Erro durante a execução de uma das etapas:", e)
    print("Verifique o traceback acima para diagnosticar o problema.")