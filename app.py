import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from PIL import Image
import os

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Simulador Forno Elétrico a Arco", layout="wide", page_icon="🏭")

# --- Barra Lateral (Configurações) ---
with st.sidebar:
    st.header("Configurações")
    
    st.subheader("1. Dados")
    uploaded_file = st.sidebar.file_uploader("Carregar base'", type=["csv"])
    
    st.divider()
    st.subheader("2. Hiperparâmetros IA")
    n_cl = st.slider("Número de Clusters", 2, 6, 3)
    max_dep = st.slider("Profundidade Árvore", 2, 20, 5)
    n_estim = st.slider("Estimadores AdaBoost", 10, 60, 30)
    
    if st.button("🔄 Re-treinar Modelos"):
        st.cache_resource.clear()

# ==========================================
# 2. TÍTULO E IMAGEM PRINCIPAL
# ==========================================
st.title("Modelo Simulador Operacional Forno Elétrico a Arco")

if os.path.exists("fea_anglo.png"):
    image = Image.open("fea_anglo.png")
    
    # Colunas para centralizar imagem
    col_img, col_vazia = st.columns([3, 1])
    
    with col_img:
        # Legenda da imagem
        st.markdown(
            """
            <div style="text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #333;">
                Esquemático do Forno Elétrico a Arco
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(image, use_container_width=True)
    
else:
    st.warning("⚠️ Imagem não encontrada no diretório.")

# ==========================================
# 3. CLASSES E FUNÇÕES (BACKEND)
# ==========================================
class Filtros:
    def nao_numerico(self, df):
        return df.select_dtypes(include=['number', 'float64', 'int64'])
    def nao_negativo(self, df):
        df = df.copy()
        num_cols = df.select_dtypes(include=['number']).columns
        df[num_cols] = df[num_cols].clip(lower=0)
        return df
    def clusterizacao(self, n_clusters, df):
        modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        indices = modelo.fit_predict(df)
        return indices, modelo

def plot_comparacao_st(nome_variavel, y_real, y_pred, n_plot):
    n_plot = min(n_plot, len(y_real))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(n_plot), y_real[:n_plot], 'ro--', markersize=4, label='Dados Reais')
    ax.plot(range(n_plot), y_pred[:n_plot], 'bo--', markersize=4, label='Modelo IA')
    ax.set_title(f"Validação: {nome_variavel}")
    ax.set_ylabel("Valor")
    ax.set_xlabel("Amostras de Teste")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

@st.cache_resource
def treinar_modelo_global(df, _max_dep, _n_estim, _n_clusters, cols_in, cols_out):
    # Treina com a base completa para uso nos simuladores
    filtro = Filtros()
    indices, kmeans = filtro.clusterizacao(_n_clusters, df[cols_in])
    
    df_proc = df.copy()
    df_proc['K classes'] = indices
    
    modelos = {}
    for i in range(_n_clusters):
        df_c = df_proc[df_proc['K classes'] == i]
        if len(df_c) < 5: continue
        
        mods_saida = {}
        for out_col in cols_out:
            regr = AdaBoostRegressor(
                DecisionTreeRegressor(max_depth=_max_dep),
                n_estimators=_n_estim, random_state=42
            )
            regr.fit(df_c[cols_in], df_c[out_col])
            mods_saida[out_col] = regr
            
        modelos[i] = mods_saida
        
    return kmeans, modelos, indices

# ==========================================
# 4. PREPARAÇÃO DE DADOS (CARREGAMENTO)
# ==========================================
# Definição das colunas baseadas na estrutura da sua base
cols_in_padrao = [f"input_{i}" for i in range(1, 40)]
cols_out_padrao = [f"output_{i}" for i in range(1, 10)]

if uploaded_file is None:
    # Gerar dados aleatórios com a estrutura correta
    total_cols = len(cols_in_padrao) + len(cols_out_padrao)
    df = pd.DataFrame(np.random.rand(200, total_cols) * 100, columns=cols_in_padrao + cols_out_padrao)
    cols_in = cols_in_padrao
    cols_out = cols_out_padrao
    st.info("ℹ️ Usando dados simulados. Faça upload do CSV para dados reais.")
else:
    try:
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='latin-1')
        for c in ["DateTime", "Potencia_Ativa_Total", "Temp. da Escória"]:
            if c in df.columns: df.drop(columns=c, inplace=True)
        
        f = Filtros()
        df = f.nao_numerico(df)
        df = f.nao_negativo(df)
        
        # Inferindo input e output pela posição
        cols_in = df.columns[:39].tolist()
        cols_out = df.columns[39:].tolist() 
        
        if len(cols_out) < 1:
             cols_out = df.columns[-9:].tolist()
             cols_in = df.columns[:-9].tolist()

        st.success(f"✅ Base Carregada: {df.shape[0]} linhas.")
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        st.stop()

# ==========================================
# 5. ABAS DA APLICAÇÃO
# ==========================================
st.divider()
tab_hist, tab_manual, tab_val = st.tabs([
    "🎛️ Simulador Manual Operacional", 
    "📈 Validação & Gráficos",
    "📋 Real vs Previsto (Cada linha)",
])

# --- TREINAMENTO GLOBAL ---
with st.spinner("Processando inteligência..."):
    kmeans_global, modelos_global, labels_global = treinar_modelo_global(
        df, max_dep, n_estim, n_cl, cols_in, cols_out
    )

# --- ABA 1: HISTÓRICO (COMPARAÇÃO COM MAPE) ---
with tab_hist:
    st.subheader("Auditoria de Dados Históricos")
    st.markdown("Selecione uma linha do passado para comparar o Real com o Previsto.")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        idx = st.number_input("Selecione o índice da linha:", 0, len(df)-1, 0)
        st.caption("Parâmetros de Entrada Reais:")
        st.dataframe(df.iloc[idx][cols_in].to_frame().T, hide_index=True)
        
        btn_check = st.button("🔍 Calcular Erro", type="primary")

    with c2:
        if btn_check:
            entrada_real = df.iloc[idx][cols_in].to_frame().T
            saida_real = df.iloc[idx][cols_out].values
            
            cluster = kmeans_global.predict(entrada_real)[0]
            st.info(f"Regime Operacional: **Cluster {cluster}**")
            
            if cluster in modelos_global:
                saida_prevista = []
                for out in cols_out:
                    val = modelos_global[cluster][out].predict(entrada_real)[0]
                    saida_prevista.append(val)
                
                # Cálculo do Erro Percentual
                saida_real_safe = np.array(saida_real)
                saida_real_safe[saida_real_safe == 0] = 0.0001 
                
                erro_percentual = np.abs((saida_real_safe - np.array(saida_prevista)) / saida_real_safe) * 100

                df_comp = pd.DataFrame({
                    "Variável": cols_out,
                    "Valor Real": saida_real,
                    "Valor Previsto (IA)": saida_prevista,
                    "Erro (%)": erro_percentual
                })
                
                st.dataframe(
                    df_comp.style.format({
                        "Valor Real": "{:.2f}", 
                        "Valor Previsto (IA)": "{:.2f}",
                        "Erro (%)": "{:.2f}%"
                    }).background_gradient(cmap="Reds", subset=["Erro (%)"]), 
                    use_container_width=True,
                    hide_index=True
                )
                
                st.metric("MAPE Médio desta linha", f"{np.mean(erro_percentual):.2f}%")
            else:
                st.warning("Cluster sem dados suficientes.")

# --- ABA 2: SIMULADOR MANUAL (PLAYGROUND) ---
with tab_manual:
    st.subheader("Simulador de Cenários")
    st.markdown("Altere os parâmetros de entrada abaixo para prever o comportamento do forno.")
    
    col_man_L, col_man_R = st.columns([1, 1])
    
    with col_man_L:
        st.write("**Ajuste os 39 Parâmetros de Entrada:**")
        input_medio = df[cols_in].mean().to_frame().T
        
        user_input = st.data_editor(
            input_medio,
            height=500,
            use_container_width=True,
            hide_index=True,
            key="editor_manual"
        )
        
        btn_sim_manual = st.button("🚀 Simular Cenário", type="primary", use_container_width=True)

    with col_man_R:
        st.write("**Saídas Previstas:**")
        if btn_sim_manual:
            cluster_man = kmeans_global.predict(user_input)[0]
            st.success(f"Regime Previsto: **Cluster {cluster_man}**")
            
            if cluster_man in modelos_global:
                preds_man = []
                for out in cols_out:
                    val = modelos_global[cluster_man][out].predict(user_input)[0]
                    preds_man.append(val)
                
                df_res_man = pd.DataFrame({
                    "Variável de Saída": cols_out,
                    "Previsão": preds_man
                })
                
                st.dataframe(
                    df_res_man.style.format({"Previsão": "{:.2f}"}).background_gradient(cmap="Blues", subset=["Previsão"]),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.error("Cluster fora da faixa de operação conhecida.")
        else:
            st.info("👈 Edite a tabela e clique em Simular.")

# --- ABA 3: VALIDAÇÃO (GRÁFICOS) ---
with tab_val:
    st.subheader("Análise de Acurácia")
    
    c_val1, c_val2 = st.columns(2)
    with c_val1:
        var_alvo = st.selectbox("Variável para Analisar:", cols_out)
    with c_val2:
        cluster_analise = st.selectbox("Filtrar por Cluster:", sorted(list(set(labels_global))))

    if st.button("Gerar Gráfico de Validação"):
        with st.spinner("Processando validação..."):
            df_temp = df.copy()
            df_temp['K'] = labels_global
            df_cluster = df_temp[df_temp['K'] == cluster_analise]
            
            if len(df_cluster) > 10:
                X = df_cluster[cols_in]
                y = df_cluster[var_alvo]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                regr_val = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_dep), n_estimators=n_estim)
                regr_val.fit(X_train, y_train)
                y_pred_val = regr_val.predict(X_test)
                
                mape = np.mean(np.abs((y_test - y_pred_val) / y_test)) * 100
                
                st.metric("Erro Médio (MAPE)", f"{mape:.2f}%")
                st.pyplot(plot_comparacao_st(var_alvo, y_test.values, y_pred_val, 100))
            else:
                st.error("Dados insuficientes neste cluster.")
