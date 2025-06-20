# Bibliotecas utilizadas no pipeline de inferência
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# Função que carrega todos os modelos e transformações já treinados
def carregar_modelos():
    modelos = {
        "scaler": joblib.load("modelos/scaler.pkl"),
        "colunas_scaler": joblib.load("modelos/colunas_scaler.pkl"),
        "encoder_model": load_model("modelos/modelo_encoder.keras"),
        "autoencoder": load_model("modelos/modelo_autoencoder.keras"),
        "kmeans": joblib.load("modelos/kmeans_auto.pkl"),
        "encoder_tipo": joblib.load("modelos/encoder_tipo_transacao.pkl"),
        "encoder_semana": joblib.load("modelos/encoder_semana.pkl"),
        "encoder_horario": joblib.load("modelos/encoder_horario.pkl")
    }
    return modelos

# Etapa de preparação dos dados antes da inferência
def preprocessar_transacoes(df: pd.DataFrame, modelos: dict) -> pd.DataFrame:
    df['transacao_data'] = pd.to_datetime(df['transacao_data'])

    dias_semana = {0:'Segunda', 1:'Terca', 2:'Quarta', 3:'Quinta', 4:'Sexta', 5:'Sabado', 6:'Domingo'}
    df['dia_de_semana'] = df['transacao_data'].dt.weekday.map(dias_semana)
    df['fim_de_semana'] = df['dia_de_semana'].isin(['Sabado', 'Domingo']).astype(int)

    def categorizar_hora(h):
        h = h.hour
        return 'Madrugada' if h < 6 else 'Manhã' if h < 12 else 'Tarde' if h < 18 else 'Noite'
    df['faixa_horaria'] = df['transacao_data'].dt.time.apply(categorizar_hora)

    df['mesma_titularidade'] = df['mesma_titularidade'].astype(int)

    # Aplica os encoders carregados para colunas categóricas
    def aplicar_encoder(df, coluna, encoder):
        arr = encoder.transform(df[[coluna]])
        nomes = encoder.get_feature_names_out([coluna])
        return pd.DataFrame(arr, columns=nomes)

    df_tipo = aplicar_encoder(df, 'transacao_tipo', modelos['encoder_tipo'])
    df_semana = aplicar_encoder(df, 'dia_de_semana', modelos['encoder_semana'])
    df_hora = aplicar_encoder(df, 'faixa_horaria', modelos['encoder_horario'])

    # Junta tudo em um único DataFrame
    df_proc = pd.concat([
        df.drop(columns=['transacao_tipo', 'dia_de_semana', 'faixa_horaria'], errors='ignore'),
        df_tipo, df_semana, df_hora
    ], axis=1)

    # Aplica normalização nas colunas numéricas
    dados_filtrados = df_proc[modelos['colunas_scaler']]
    df_proc[modelos['colunas_scaler']] = modelos['scaler'].transform(dados_filtrados)

    return df_proc

# Usa o autoencoder para calcular desvios e o KMeans para agrupar padrões de comportamento
def detectar_anomalias(df_proc, modelos):
    reconstruido = modelos['autoencoder'].predict(df_proc[modelos['colunas_scaler']])
    erro_reconstrucao = np.mean(np.square(df_proc[modelos['colunas_scaler']] - reconstruido), axis=1)

    dados_latentes = modelos['encoder_model'].predict(df_proc[modelos['colunas_scaler']])
    labels = modelos['kmeans'].predict(dados_latentes)

    # Classifica a suspeita de acordo com o erro de reconstrução
    q75, q90, q95 = np.quantile(erro_reconstrucao, [0.75, 0.90, 0.95])
    condicoes = [erro_reconstrucao > q95, erro_reconstrucao > q90, erro_reconstrucao > q75]
    valores = ['alta', 'media', 'baixa']
    suspeita = np.select(condicoes, valores, default='nenhuma')

    df_proc['erro_reconstrucao'] = erro_reconstrucao
    df_proc['cluster_autoencoder'] = labels
    df_proc['suspeita'] = suspeita

    # Agora avalia a suspeita considerando o comportamento do cluster
    thresholds_cluster = df_proc.groupby("cluster_autoencoder")['erro_reconstrucao'].quantile(0.95).to_dict()
    
    def classificar_suspeita_cluster(row):
        threshold = thresholds_cluster[row['cluster_autoencoder']]
        erro = row['erro_reconstrucao']
        if erro > threshold * 1.5:
            return "alta"
        elif erro > threshold:
            return "media"
        elif erro > threshold * 0.5:
            return "baixa"
        else:
            return "nenhuma"

    df_proc['suspeita_cluster'] = df_proc.apply(classificar_suspeita_cluster, axis=1)
    return df_proc

# Gera o perfil médio de transações por conta com base no histórico
def gerar_perfil_cliente(df_conta):
    perfil = {
        'media_valor': df_conta['transacao_valor'].mean(),
        'std_valor': df_conta['transacao_valor'].std(),
        'percentual_pix': df_conta.filter(like='transacao_tipo_pix').sum(axis=1).mean(),
        'percentual_transferencia': df_conta.filter(like='transacao_tipo_transferencia').sum(axis=1).mean(),
        'percentual_pagamento': df_conta.filter(like='transacao_tipo_pagamento').sum(axis=1).mean(),
        'percentual_saque': df_conta.filter(like='transacao_tipo_saque').sum(axis=1).mean(),
        'percentual_deposito': df_conta.filter(like='transacao_tipo_deposito').sum(axis=1).mean(),
        'percentual_fim_de_semana': df_conta['fim_de_semana'].mean(),
        'percentual_mesma_titularidade': df_conta['mesma_titularidade'].mean()
    }
    # Horário e dia da semana mais comum
    horario_cols = [c for c in df_conta.columns if c.startswith('faixa_horaria_')]
    if horario_cols:
        perfil['horario_mais_comum'] = df_conta[horario_cols].sum().idxmax().replace('faixa_horaria_', '')
    dia_cols = [c for c in df_conta.columns if c.startswith('dia_de_semana_')]
    if dia_cols:
        perfil['dia_semana_mais_comum'] = df_conta[dia_cols].sum().idxmax().replace('dia_de_semana_', '')
    return pd.Series(perfil)

# Aplica o perfil para cada conta presente no conjunto de dados
def gerar_perfis(df_final):
    return df_final.groupby('conta_id', group_keys=False).apply(gerar_perfil_cliente).reset_index()

# Função principal que orquestra todas as etapas de inferência
def rodar_inferencia(path_csv: str):
    print("Carregando modelos...")
    modelos = carregar_modelos()

    print(f"Lendo dados de: {path_csv}")
    df = pd.read_csv(path_csv)

    print("Pré-processando transações...")
    df_proc = preprocessar_transacoes(df.copy(), modelos)

    print("🔎 Avaliando transações fora do padrão...")
    df_final = detectar_anomalias(df_proc, modelos)

    print("Gerando perfis por conta...")
    perfis = gerar_perfis(df_final)

    print("Juntando dados...")
    df_completo = pd.merge(df_final, perfis, on="conta_id", how="left")

    print("Salvando CSV final...")
    os.makedirs("resultados", exist_ok=True)
    df_completo.to_csv("resultados/transacoes_com_comportamento_por_conta.csv", index=False)
    print("✅ Inferência concluída.")
    print("\n🔎 Contas únicas no conjunto de transações:")
    print(df_completo['conta_id'].nunique())
    print("\n🧾 Lista de contas (amostra):")
    print(df_completo['conta_id'].unique()[:10])

# Permite rodar tudo direto pelo terminal
if __name__ == "__main__":
    rodar_inferencia("dados/transacoes_final_fraude.csv")





