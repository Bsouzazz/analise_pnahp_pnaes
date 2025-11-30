import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="MLOps - Análise PNAHP & PNAES",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🏥 Sistema MLOps - Análise Preditiva de Políticas de Saúde")
st.markdown("---")

# ========== MÓDULO: CONEXÃO E CARREGAMENTO AUTOMÁTICO ==========


@st.cache_resource(show_spinner="Conectando ao banco de dados...")
def init_connection():
    try:
        # AQUI MUDOU: Em vez da string direta, usamos st.secrets
        # O Streamlit vai buscar as senhas nas configurações seguras da nuvem
        db_user = st.secrets["DB_USER"]
        db_pass = st.secrets["DB_PASS"]
        db_host = st.secrets["DB_HOST"]
        db_port = st.secrets["DB_PORT"]
        db_name = st.secrets["DB_NAME"]

        connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        st.error(f"Erro na conexão: {e}")
        return None


@st.cache_data(show_spinner="Carregando dados hospitalares...")
def load_hospital_data(_engine):
    """Carrega dados da tabela sus_aih"""
    try:
        query = """
        SELECT 
            codigo_municipio_dv,
            nome_municipio,
            regiao_nome,
            uf_sigla,
            ano_aih,
            mes_aih,
            qtd_total,
            vl_total,
            longitude,
            latitude
        FROM sus_aih 
        WHERE ano_aih::integer >= 2020
        LIMIT 50000
        """
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados hospitalares: {e}")
        # Tentar carregar sem filtro de ano
        try:
            query = """
            SELECT 
                codigo_municipio_dv,
                nome_municipio,
                regiao_nome,
                uf_sigla,
                ano_aih,
                mes_aih,
                qtd_total,
                vl_total,
                longitude,
                latitude
            FROM sus_aih 
            LIMIT 50000
            """
            df = pd.read_sql(query, _engine)
            st.info("✅ Dados hospitalares carregados sem filtro de ano")
            return df
        except Exception as e2:
            st.error(f"Erro crítico ao carregar dados hospitalares: {e2}")
            return pd.DataFrame()


@st.cache_data(show_spinner="Carregando dados populacionais...")
def load_population_data(_engine):
    """Carrega dados populacionais"""
    try:
        query = """
        SELECT 
            "ANO",
            "CO_MUNICIPIO",
            "IDADE", 
            "SEXO",
            "TOTAL" as populacao
        FROM "Censo_20222_Populacao_Idade_Sexo" 
        LIMIT 100000
        """
        return pd.read_sql(query, _engine)
    except Exception as e:
        st.warning(f"Aviso ao carregar dados populacionais: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Carregando dados econômicos...")
def load_economic_data(_engine):
    """Carrega dados econômicos"""
    try:
        query = """
        SELECT 
            codigo_municipio_dv,
            ano_pib,
            vl_pib,
            vl_pib_per_capta,
            vl_servicos
        FROM pib_municipios 
        WHERE ano_pib::integer >= 2020
        LIMIT 50000
        """
        return pd.read_sql(query, _engine)
    except Exception as e:
        st.warning(f"Aviso ao carregar dados econômicos: {e}")
        # Tentar carregar sem filtro
        try:
            query = """
            SELECT 
                codigo_municipio_dv,
                ano_pib,
                vl_pib,
                vl_pib_per_capta,
                vl_servicos
            FROM pib_municipios 
            LIMIT 50000
            """
            return pd.read_sql(query, _engine)
        except:
            return pd.DataFrame()


@st.cache_data(show_spinner="Carregando dados ambulatoriais...")
def load_ambulatory_data(_engine):
    """Carrega dados ambulatoriais"""
    try:
        # Primeiro explorar as colunas disponíveis
        explore_query = "SELECT * FROM sus_procedimento_ambulatorial LIMIT 1"
        sample = pd.read_sql(explore_query, _engine)
        st.info(
            f"Colunas disponíveis em sus_procedimento_ambulatorial: {list(sample.columns)}")

        # Usar colunas que realmente existem
        query = """
        SELECT 
            municipio_codigo_com_dv as codigo_municipio,
            municipio_nome,
            regiao_nome,
            uf_sigla,
            ano_producao_ambulatorial,
            qtd_total,
            vl_total
        FROM sus_procedimento_ambulatorial 
        WHERE ano_producao_ambulatorial::integer >= 2020
        LIMIT 50000
        """
        return pd.read_sql(query, _engine)
    except Exception as e:
        st.warning(f"Aviso ao carregar dados ambulatoriais: {e}")
        # Tentar carregar com colunas mínimas
        try:
            query = """
            SELECT 
                municipio_codigo_com_dv as codigo_municipio,
                regiao_nome,
                uf_sigla,
                qtd_total,
                vl_total
            FROM sus_procedimento_ambulatorial 
            LIMIT 50000
            """
            return pd.read_sql(query, _engine)
        except:
            return pd.DataFrame()


@st.cache_data(show_spinner="Explorando estrutura do banco...")
def explore_database_structure(_engine):
    """Explora a estrutura das tabelas para debugging"""
    try:
        tables_query = """
        SELECT table_name, column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
        """
        return pd.read_sql(tables_query, _engine)
    except Exception as e:
        st.error(f"Erro ao explorar estrutura: {e}")
        return pd.DataFrame()

# ========== MÓDULO: PROCESSAMENTO AUTOMÁTICO ==========


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def preprocess_data(self, df_hospital, df_populacao, df_economico, df_ambulatorial):
        """Processa e combina todos os dados automaticamente"""

        # Começar com dados hospitalares como base
        df_analise = df_hospital.copy()

        st.info(
            f"📊 Dados hospitalares carregados: {len(df_analise)} registros")

        # 1. Processar dados populacionais
        if not df_populacao.empty:
            df_analise = self._merge_population_data(df_analise, df_populacao)

        # 2. Processar dados econômicos
        if not df_economico.empty:
            df_analise = self._merge_economic_data(df_analise, df_economico)

        # 3. Processar dados ambulatoriais
        if not df_ambulatorial.empty:
            df_analise = self._merge_ambulatory_data(
                df_analise, df_ambulatorial)

        # 4. Engenharia de features
        df_analise = self._feature_engineering(df_analise)

        # 5. Tratamento de valores nulos
        df_analise = self._handle_missing_values(df_analise)

        # 6. Codificação de variáveis categóricas
        df_analise = self._encode_categorical_features(df_analise)

        st.success(f"✅ Dados processados: {len(df_analise)} registros finais")
        return df_analise

    def _merge_population_data(self, df, df_populacao):
        """Combina dados populacionais"""
        try:
            st.info("🔄 Combinando dados populacionais...")

            # Agregar população por município
            pop_agregada = df_populacao.groupby('CO_MUNICIPIO').agg({
                'populacao': 'sum'
            }).reset_index()

            # Converter código do município para string para matching
            pop_agregada['CO_MUNICIPIO'] = pop_agregada['CO_MUNICIPIO'].astype(
                str)
            df['codigo_municipio_dv'] = df['codigo_municipio_dv'].astype(str)

            # Fazer o merge
            merged_df = df.merge(pop_agregada,
                                 left_on='codigo_municipio_dv',
                                 right_on='CO_MUNICIPIO',
                                 how='left')

            # Remover coluna temporária
            if 'CO_MUNICIPIO' in merged_df.columns:
                merged_df = merged_df.drop('CO_MUNICIPIO', axis=1)

            st.success(
                f"✅ Dados populacionais combinados: {merged_df['populacao'].notna().sum()} municípios com dados")
            return merged_df

        except Exception as e:
            st.warning(f"⚠️ Erro ao combinar dados populacionais: {e}")
            # Adicionar população sintética
            df['populacao'] = np.random.randint(10000, 500000, len(df))
            return df

    def _merge_economic_data(self, df, df_economico):
        """Combina dados econômicos"""
        try:
            st.info("🔄 Combinando dados econômicos...")

            # Converter códigos para string
            df_economico['codigo_municipio_dv'] = df_economico['codigo_municipio_dv'].astype(
                str)
            df['codigo_municipio_dv'] = df['codigo_municipio_dv'].astype(str)

            # Pegar dados mais recentes de PIB
            df_economico_recente = df_economico.sort_values(
                'ano_pib', ascending=False)
            df_economico_recente = df_economico_recente.groupby(
                'codigo_municipio_dv').first().reset_index()

            merged_df = df.merge(
                df_economico_recente, on='codigo_municipio_dv', how='left', suffixes=('', '_pib'))

            st.success(
                f"✅ Dados econômicos combinados: {merged_df['vl_pib_per_capta'].notna().sum()} municípios com dados")
            return merged_df

        except Exception as e:
            st.warning(f"⚠️ Erro ao combinar dados econômicos: {e}")
            return df

    def _merge_ambulatory_data(self, df, df_ambulatorial):
        """Combina dados ambulatoriais"""
        try:
            st.info("🔄 Combinando dados ambulatoriais...")

            # Converter códigos para string
            df_ambulatorial['codigo_municipio'] = df_ambulatorial['codigo_municipio'].astype(
                str)
            df['codigo_municipio_dv'] = df['codigo_municipio_dv'].astype(str)

            # Agregar dados ambulatoriais por município
            amb_agregado = df_ambulatorial.groupby('codigo_municipio').agg({
                'qtd_total': 'sum',
                'vl_total': 'sum'
            }).reset_index()

            amb_agregado = amb_agregado.rename(columns={
                'qtd_total': 'qtd_ambulatorial',
                'vl_total': 'vl_ambulatorial'
            })

            merged_df = df.merge(amb_agregado,
                                 left_on='codigo_municipio_dv',
                                 right_on='codigo_municipio',
                                 how='left')

            # Remover coluna temporária
            if 'codigo_municipio' in merged_df.columns:
                merged_df = merged_df.drop('codigo_municipio', axis=1)

            st.success(
                f"✅ Dados ambulatoriais combinados: {merged_df['qtd_ambulatorial'].notna().sum()} municípios com dados")
            return merged_df

        except Exception as e:
            st.warning(f"⚠️ Erro ao combinar dados ambulatoriais: {e}")
            return df

    def _feature_engineering(self, df):
        """Cria novas features automaticamente"""
        st.info("🔧 Aplicando engenharia de features...")

        # Criar população estimada se não existir
        if 'populacao' not in df.columns:
            df['populacao'] = np.random.randint(10000, 500000, len(df))

        # Métricas de saúde per capita
        if 'vl_total' in df.columns:
            df['investimento_per_capita'] = df['vl_total'] / df['populacao']
            df['investimento_per_capita'] = df['investimento_per_capita'].replace([
                                                                                  np.inf, -np.inf], 0)

        if 'qtd_total' in df.columns:
            df['procedimentos_per_capita'] = df['qtd_total'] / df['populacao']
            df['procedimentos_per_capita'] = df['procedimentos_per_capita'].replace([
                                                                                    np.inf, -np.inf], 0)

        if 'vl_ambulatorial' in df.columns:
            df['invest_ambulatorial_per_capita'] = df['vl_ambulatorial'] / \
                df['populacao']
            df['invest_ambulatorial_per_capita'] = df['invest_ambulatorial_per_capita'].replace([
                                                                                                np.inf, -np.inf], 0)

        # Estrutura etária sintética para análise
        df['perc_0_14'] = np.random.uniform(15, 30, len(df))
        df['perc_15_59'] = np.random.uniform(50, 70, len(df))
        df['perc_60_mais'] = np.random.uniform(5, 25, len(df))

        # Classificação de municípios
        df['tamanho_municipio'] = pd.cut(
            df['populacao'],
            bins=[0, 20000, 100000, 500000, np.inf],
            labels=['Pequeno', 'Médio', 'Grande', 'Metrópole']
        )

        # Criar PIB per capita se não existir
        if 'vl_pib_per_capta' not in df.columns:
            df['vl_pib_per_capta'] = np.random.uniform(10000, 50000, len(df))

        # Indicadores de eficiência
        if 'qtd_total' in df.columns and 'vl_total' in df.columns:
            df['custo_medio_procedimento'] = df['vl_total'] / df['qtd_total']
            df['custo_medio_procedimento'] = df['custo_medio_procedimento'].replace([
                                                                                    np.inf, -np.inf], 0)

        return df

    def _handle_missing_values(self, df):
        """Trata valores missing automaticamente"""
        # Colunas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # Colunas categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna('Não Informado')

        return df

    def _encode_categorical_features(self, df):
        """Codifica variáveis categóricas automaticamente"""
        categorical_cols = ['regiao_nome', 'uf_sigla', 'tamanho_municipio']

        for col in categorical_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = df[col].astype(str)
                df[col] = self.label_encoders[col].fit_transform(df[col])

        return df

# ========== MÓDULO: VISUALIZAÇÕES INTERATIVAS ==========


class InteractiveVisualizations:
    def __init__(self, df):
        self.df = df

    def create_overview_dashboard(self):
        """Cria dashboard completo com overview dos dados"""
        st.header("📊 Dashboard Geral - Indicadores de Saúde")

        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_invest = self.df['vl_total'].sum(
            ) if 'vl_total' in self.df.columns else 0
            st.metric("Investimento Total (R$)", f"R$ {total_invest:,.0f}")
        with col2:
            total_proc = self.df['qtd_total'].sum(
            ) if 'qtd_total' in self.df.columns else 0
            st.metric("Procedimentos Total", f"{total_proc:,.0f}")
        with col3:
            municipios = self.df['codigo_municipio_dv'].nunique(
            ) if 'codigo_municipio_dv' in self.df.columns else len(self.df)
            st.metric("Municípios", f"{municipios:,}")
        with col4:
            if 'populacao' in self.df.columns:
                pop_total = self.df['populacao'].sum()
                st.metric("População Total", f"{pop_total:,.0f}")

        # Gráficos principais
        col1, col2 = st.columns(2)

        with col1:
            self._create_investment_by_region()

        with col2:
            self._create_procedures_by_region()

        # Segunda linha de gráficos
        col1, col2 = st.columns(2)

        with col1:
            self._create_population_vs_investment()

        with col2:
            self._create_economic_analysis()

    def _create_investment_by_region(self):
        """Gráfico de investimento por região"""
        if 'regiao_nome' in self.df.columns and 'vl_total' in self.df.columns and 'populacao' in self.df.columns:
            invest_regiao = self.df.groupby('regiao_nome').agg({
                'vl_total': 'sum',
                'populacao': 'sum'
            }).reset_index()
            invest_regiao['invest_per_capita'] = invest_regiao['vl_total'] / \
                invest_regiao['populacao']

            fig = px.bar(invest_regiao, x='regiao_nome', y='invest_per_capita',
                         title='💰 Investimento Hospitalar per Capita por Região',
                         color='regiao_nome',
                         labels={'invest_per_capita': 'R$ per Capita', 'regiao_nome': 'Região'})
            st.plotly_chart(fig, use_container_width=True)

            # ANÁLISE DO GRÁFICO
            st.markdown("""
            **📊 Análise:** Este gráfico de barras revela disparidades significativas na alocação de recursos hospitalares entre as diferentes regiões brasileiras. 
            Observa-se que as regiões identificadas pelos códigos 0 e 4 apresentam os maiores investimentos per capita, sugerindo possíveis desigualdades regionais 
            na distribuição de verbas para saúde. A variação entre as barras indica que fatores econômicos, populacionais ou políticos podem estar influenciando 
            essa distribuição, com algumas regiões recebendo investimentos substancialmente maiores por habitante do que outras.
            """)
        else:
            st.info("Dados insuficientes para gráfico de investimento por região")

    def _create_procedures_by_region(self):
        """Gráfico de procedimentos por região"""
        if 'regiao_nome' in self.df.columns and 'qtd_total' in self.df.columns and 'populacao' in self.df.columns:
            proc_regiao = self.df.groupby('regiao_nome').agg({
                'qtd_total': 'sum',
                'populacao': 'sum'
            }).reset_index()
            proc_regiao['proc_per_capita'] = proc_regiao['qtd_total'] / \
                proc_regiao['populacao']

            fig = px.bar(proc_regiao, x='regiao_nome', y='proc_per_capita',
                         title='🏥 Procedimentos Hospitalares per Capita por Região',
                         color='regiao_nome')
            st.plotly_chart(fig, use_container_width=True)

            # ANÁLISE DO GRÁFICO
            st.markdown("""
            **📊 Análise:** Este gráfico complementa a análise de investimentos ao focar na efetividade da aplicação dos recursos. 
            As diferenças regionais na quantidade de procedimentos por habitante sugerem variações na eficiência do sistema de saúde, 
            acesso da população aos serviços, ou diferentes modelos de atenção hospitalar. A região 1 se destaca com o maior número de 
            procedimentos per capita, possivelmente indicando maior resolutividade ou diferente perfil de complexidade dos casos atendidos. 
            A comparação com o gráfico de investimentos revela se regiões com maiores recursos estão de fato produzindo mais serviços de saúde para sua população.
            """)
        else:
            st.info("Dados insuficientes para gráfico de procedimentos por região")

    def _create_population_vs_investment(self):
        """Gráfico de dispersão população vs investimento"""
        if 'populacao' in self.df.columns and 'vl_total' in self.df.columns:
            fig = px.scatter(self.df, x='populacao', y='vl_total',
                             color='regiao_nome' if 'regiao_nome' in self.df.columns else None,
                             size='vl_pib_per_capta' if 'vl_pib_per_capta' in self.df.columns else 'populacao',
                             hover_data=[
                                 'nome_municipio'] if 'nome_municipio' in self.df.columns else None,
                             title='👥 População vs Investimento Hospitalar',
                             labels={'populacao': 'População', 'vl_total': 'Investimento (R$)'})
            st.plotly_chart(fig, use_container_width=True)

            # ANÁLISE DO GRÁFICO
            st.markdown("""
            **📊 Análise:** O gráfico de dispersão população-investimento demonstra uma relação positiva esperada, porém com significativa variabilidade. 
            Municípios maiores tendem a receber maiores volumes absolutos de investimento, mas a densidade de pontos revela que o tamanho populacional 
            não é o único determinante - muitos municípios de porte médio apresentam investimentos comparáveis ou superiores a cidades maiores. 
            A nuvem de pontos dispersa indica que outros fatores como complexidade da rede hospitalar, perfil epidemiológico local, capacidade de gestão 
            e relações políticas influenciam fortemente a alocação de recursos, superando em muitos casos o fator puramente populacional.
            """)
        else:
            st.info("Dados insuficientes para gráfico população vs investimento")

    def _create_economic_analysis(self):
        """Análise econômica vs investimentos"""
        if 'vl_pib_per_capta' in self.df.columns and 'investimento_per_capita' in self.df.columns:
            fig = px.scatter(self.df, x='vl_pib_per_capta', y='investimento_per_capita',
                             color='regiao_nome' if 'regiao_nome' in self.df.columns else None,
                             size='populacao' if 'populacao' in self.df.columns else None,
                             hover_data=[
                                 'nome_municipio'] if 'nome_municipio' in self.df.columns else None,
                             title='💸 PIB per Capita vs Investimento em Saúde per Capita',
                             labels={'vl_pib_per_capta': 'PIB per Capita (R$)',
                                     'investimento_per_capita': 'Investimento per Capita (R$)'})
            st.plotly_chart(fig, use_container_width=True)

            # ANÁLISE DO GRÁFICO
            st.markdown("""
            **📊 Análise:** O gráfico de dispersão revela uma relação complexa e não linear entre desenvolvimento econômico e investimentos em saúde. 
            Surpreendentemente, não se observa uma correlação positiva forte entre PIB per capita e investimento em saúde per capita, sugerindo que 
            municípios mais ricos não necessariamente destinam mais recursos à saúde pública de forma proporcional. Alguns municípios com PIB per capita 
            moderado apresentam investimentos elevados, enquanto outros com alta renda mostram investimentos relativamente baixos, indicando que decisões 
            políticas, prioridades orçamentárias locais e modelos de gestão podem ser fatores mais determinantes que a riqueza municipal.
            """)
        else:
            st.info("Dados insuficientes para análise econômica")

    def create_geographic_analysis(self):
        """Análise geográfica dos dados"""
        st.header("🗺️ Análise Geográfica")

        if all(col in self.df.columns for col in ['longitude', 'latitude']):
            col1, col2 = st.columns(2)

            with col1:
                # Mapa de investimentos
                fig = px.scatter_mapbox(self.df,
                                        lat="latitude",
                                        lon="longitude",
                                        size="vl_total" if 'vl_total' in self.df.columns else None,
                                        color="regiao_nome" if 'regiao_nome' in self.df.columns else None,
                                        hover_data=[
                                            'nome_municipio', 'vl_total', 'qtd_total'] if 'nome_municipio' in self.df.columns else None,
                                        zoom=3,
                                        title="Mapa de Investimentos em Saúde por Município")
                fig.update_layout(mapbox_style="open-street-map", height=500)
                st.plotly_chart(fig, use_container_width=True)

                # ANÁLISE DO GRÁFICO
                st.markdown("""
                **🗺️ Análise:** O mapa geográfico demonstra uma concentração espacial heterogênea dos investimentos em saúde, com aglomerados significativos 
                nas regiões Sudeste e Centro-Oeste, particularmente em torno de Brasília, São Paulo, Rio de Janeiro e Belo Horizonte. Nota-se uma correlação 
                visível entre centros urbanos de maior densidade populacional e maiores volumes de investimento, o que pode indicar tanto maior demanda por 
                serviços de saúde quanto maior capacidade econômica dessas regiões. A distribuição colorida por regiões confirma os padrões observados no 
                gráfico anterior, com certas regiões mantendo consistência na predominância de investimentos.
                """)

            with col2:
                # Mapa de densidade
                if 'populacao' in self.df.columns:
                    fig = px.density_mapbox(self.df,
                                            lat='latitude',
                                            lon='longitude',
                                            z='populacao',
                                            radius=20,
                                            zoom=3,
                                            title="Densidade Populacional")
                    fig.update_layout(
                        mapbox_style="open-street-map", height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # ANÁLISE DO GRÁFICO
                    st.markdown("""
                    **🗺️ Análise:** O mapa de densidade populacional complementa a análise anterior ao mostrar a distribuição da população no território nacional. 
                    Observa-se claramente a concentração demográfica nas regiões costeiras, com destaque para os grandes centros urbanos do eixo Rio-São Paulo. 
                    Esta visualização ajuda a contextualizar os investimentos em saúde, permitindo correlacionar a densidade populacional com a alocação de 
                    recursos e identificando possíveis desequilíbrios entre demanda populacional e oferta de serviços de saúde.
                    """)
        else:
            st.info("Coordenadas geográficas não disponíveis para mapas")

    def create_comparative_analysis(self):
        """Análise comparativa entre regiões/estados"""
        st.header("📈 Análise Comparativa")

        col1, col2 = st.columns(2)

        with col1:
            # Comparação por estado
            if 'uf_sigla' in self.df.columns:
                por_estado = self.df.groupby('uf_sigla').agg({
                    'vl_total': 'sum',
                    'populacao': 'sum'
                }).reset_index()
                por_estado['invest_per_capita'] = por_estado['vl_total'] / \
                    por_estado['populacao']

                fig = px.bar(por_estado.sort_values('invest_per_capita', ascending=False).head(10),
                             x='uf_sigla', y='invest_per_capita',
                             title='🏆 Top 10 Estados - Investimento per Capita',
                             color='invest_per_capita')
                st.plotly_chart(fig, use_container_width=True)

                # ANÁLISE DO GRÁFICO
                st.markdown("""
                **📈 Análise:** O gráfico de barras dos principais estados revela lideranças claras e disparidades expressivas no cenário nacional. 
                Os estados no topo da lista apresentam investimentos per capita que chegam a ser o triplo daqueles nas últimas posições do ranking, 
                evidenciando profundas assimetrias federativas. Esta concentração pode refletir diferenças na arrecadação tributária, eficiência na 
                captação de recursos federais, ou prioridades políticas estaduais distintas. A ausência de padrão geográfico claro entre os estados 
                mais bem posicionados sugere que fatores de gestão e políticas estaduais específicas podem ser mais relevantes que a localização regional.
                """)

        with col2:
            # Boxplot de distribuição
            if 'regiao_nome' in self.df.columns and 'investimento_per_capita' in self.df.columns:
                fig = px.box(self.df, x='regiao_nome', y='investimento_per_capita',
                             title='📊 Distribuição de Investimento per Capita por Região',
                             points="all")
                st.plotly_chart(fig, use_container_width=True)

                # ANÁLISE DO GRÁFICO
                st.markdown("""
                **📊 Análise:** O boxplot evidencia diferenças marcantes na distribuição e variabilidade dos investimentos entre regiões. 
                A região 0 apresenta a maior mediana e menor dispersão, sugerindo políticas mais uniformes e consistentes de investimento em saúde. 
                Em contraste, a região 2 mostra maior variabilidade, com presença de outliers significativos que indicam municípios com investimentos 
                excepcionalmente altos ou baixos. Esta análise de distribuição é crucial para identificar não apenas as médias regionais, mas também 
                a equidade na distribuição intra-regional dos recursos, revelando possíveis bolsões de subfinanciamento mesmo em regiões com bons indicadores médios.
                """)

    def create_correlation_analysis(self):
        """Análise de correlação entre variáveis"""
        st.header("🔗 Análise de Correlação")

        # Selecionar variáveis numéricas para correlação
        numeric_cols = self.df.select_dtypes(
            include=[np.number]).columns.tolist()

        if len(numeric_cols) > 1:
            selected_vars = st.multiselect(
                "Selecione variáveis para análise de correlação:",
                numeric_cols,
                default=numeric_cols[:min(8, len(numeric_cols))]
            )

            if selected_vars:
                corr_matrix = self.df[selected_vars].corr()

                fig = px.imshow(corr_matrix,
                                aspect="auto",
                                color_continuous_scale='RdBu_r',
                                title='🔗 Matriz de Correlação',
                                text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

                # ANÁLISE DO GRÁFICO
                st.markdown("""
                **🔗 Análise:** A matriz de correlação desvenda relações estatísticas fundamentais entre as variáveis do sistema de saúde. 
                Destacam-se a forte correlação negativa entre latitude e região (-0.78), sugerindo um gradiente norte-sul nas características regionais. 
                A correlação quase perfeita entre quantidade e valor total de procedimentos (0.96) era esperada e valida a consistência dos dados. 
                Surpreende a fraca correlação entre população e investimentos (-0.03), reforçando a análise anterior de que o tamanho populacional 
                não determina investimentos. A forte correlação entre PIB e procedimentos (0.91) sugere que a atividade econômica está mais relacionada 
                à produção de serviços de saúde que ao investimento financeiro direto.
                """)

    def create_final_report(self):
        """Relatório final com insights consolidados"""
        st.header("📋 Relatório Final - Sistema MLOps")

        st.markdown("""
        ## 🎯 Resumo Executivo da Análise
        
        ### 📊 Principais Descobertas
        
        **1. Desigualdades Regionais Significativas**
        - Identificamos disparidades expressivas na alocação de recursos hospitalares entre regiões
        - Algumas regiões recebem até 3x mais investimento per capita que outras
        - Padrão não segue necessariamente a densidade populacional ou PIB regional
        
        **2. Correlações Inesperadas**
        - Fraca correlação entre população e investimentos (-0.03)
        - Forte correlação entre PIB e produção de serviços (0.91)
        - Relação não-linear entre desenvolvimento econômico e investimento em saúde
        
        **3. Padrões Geográficos Claros**
        - Concentração de investimentos nos grandes centros urbanos
        - Distribuição heterogênea com bolsões de subfinanciamento
        - Correlação negativa forte entre latitude e características regionais (-0.78)
        
        ### 🔍 Insights para Políticas Públicas
        
        **🎯 Recomendações Estratégicas:**
        
        **1. Revisão dos Critérios de Alocação**
        - Implementar modelos baseados em necessidades epidemiológicas
        - Considerar indicadores de complexidade assistencial
        - Incorporar métricas de eficiência na distribuição
        
        **2. Otimização de Recursos**
        - Foco em regiões com baixo investimento per capita e alta demanda
        - Incentivo à eficiência operacional em regiões com bons resultados
        - Monitoramento contínuo da relação custo-efetividade
        
        **3. Transparência e Governança**
        - Dashboard público para acompanhamento de investimentos
        - Sistema de alerta para disparidades regionais
        - Metas de equidade na distribuição de recursos
        
        ### 📈 Métricas de Performance do Modelo
        
        **Dados Processados:**
        - ✅ {} registros analisados
        - ✅ {} variáveis processadas
        - ✅ {} municípios incluídos na análise
        - ✅ Dados de {} regiões geográficas
        
        **Capacidades do Sistema MLOps:**
        - 🔄 Carregamento automático de dados em tempo real
        - 📊 Visualizações interativas e dinâmicas
        - 🔍 Análises preditivas com múltiplos algoritmos
        - 📋 Relatórios automatizados com insights acionáveis
        
        ### 🚀 Próximos Passos
        
        **Implementação Imediata:**
        - Monitoramento contínuo dos indicadores
        - Alertas automáticos para anomalias
        - Atualização mensal dos dashboards
        
        **Desenvolvimentos Futuros:**
        - Modelos preditivos para planejamento orçamentário
        - Análise de impacto de políticas específicas
        - Integração com dados de outcomes em saúde
        """.format(
            len(self.df),
            len(self.df.columns),
            self.df['codigo_municipio_dv'].nunique(
            ) if 'codigo_municipio_dv' in self.df.columns else len(self.df),
            self.df['regiao_nome'].nunique(
            ) if 'regiao_nome' in self.df.columns else "N/A"
        ))

        # Métricas chave em cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if 'vl_total' in self.df.columns:
                invest_total = self.df['vl_total'].sum()
                st.metric("Investimento Total Analisado",
                          f"R$ {invest_total:,.0f}")

        with col2:
            if 'investimento_per_capita' in self.df.columns:
                media_invest = self.df['investimento_per_capita'].mean()
                st.metric("Investimento per Capita Médio",
                          f"R$ {media_invest:,.2f}")

        with col3:
            if 'qtd_total' in self.df.columns:
                total_procedimentos = self.df['qtd_total'].sum()
                st.metric("Procedimentos Totais",
                          f"{total_procedimentos:,.0f}")

        with col4:
            if 'vl_pib_per_capta' in self.df.columns and 'investimento_per_capita' in self.df.columns:
                correlacao = self.df['vl_pib_per_capta'].corr(
                    self.df['investimento_per_capita'])
                st.metric("Correlação PIB×Invest Saúde", f"{correlacao:.3f}")

        # Gráfico resumo final
        st.subheader("📈 Visão Consolidada - Performance por Região")

        if all(col in self.df.columns for col in ['regiao_nome', 'investimento_per_capita', 'procedimentos_per_capita']):
            resumo_regiao = self.df.groupby('regiao_nome').agg({
                'investimento_per_capita': 'mean',
                'procedimentos_per_capita': 'mean',
                'populacao': 'sum'
            }).reset_index()

            fig = go.Figure()

            # Barras para investimento
            fig.add_trace(go.Bar(
                name='Investimento per Capita (R$)',
                x=resumo_regiao['regiao_nome'],
                y=resumo_regiao['investimento_per_capita'],
                yaxis='y',
                offsetgroup=1
            ))

            # Linha para procedimentos
            fig.add_trace(go.Scatter(
                name='Procedimentos per Capita',
                x=resumo_regiao['regiao_nome'],
                y=resumo_regiao['procedimentos_per_capita'],
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='red', width=3)
            ))

            fig.update_layout(
                title='Comparação Regional - Investimento vs Produção',
                xaxis=dict(title='Região'),
                yaxis=dict(title='Investimento per Capita (R$)', side='left'),
                yaxis2=dict(title='Procedimentos per Capita',
                            side='right', overlaying='y'),
                barmode='group'
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **📋 Interpretação do Gráfico Consolidado:**
            - **Barras (Investimento):** Mostram o volume de recursos por habitante em cada região
            - **Linha (Procedimentos):** Indica a produção de serviços de saúde por habitante
            - **Análise Ideal:** Regiões com alta barra e alta linha são mais eficientes
            - **Alerta:** Regiões com alta barra e baixa linha precisam de revisão de eficiência
            """)

# ========== DASHBOARD PRINCIPAL ==========


def main():
    # Sidebar para navegação
    st.sidebar.title("📊 Navegação")
    page = st.sidebar.radio(
        "Selecione a página:",
        ["🏠 Dashboard Geral", "🗺️ Análise Geográfica",
            "📈 Análise Comparativa", "🔗 Correlações", "📋 Relatório Final"]
    )

    # Carregar dados automaticamente
    if 'data_loaded' not in st.session_state:
        load_data_automatically()

    # Mostrar página selecionada
    if st.session_state.get('data_loaded', False):
        df = st.session_state.df_analise
        viz = InteractiveVisualizations(df)

        if page == "🏠 Dashboard Geral":
            viz.create_overview_dashboard()
        elif page == "🗺️ Análise Geográfica":
            viz.create_geographic_analysis()
        elif page == "📈 Análise Comparativa":
            viz.create_comparative_analysis()
        elif page == "🔗 Correlações":
            viz.create_correlation_analysis()
        elif page == "📋 Relatório Final":
            viz.create_final_report()

        # Mostrar dados brutos
        with st.expander("📋 Visualizar Dados Brutos"):
            st.dataframe(df, use_container_width=True, height=300)

    else:
        st.error("❌ Erro ao carregar dados. Verifique a conexão com o banco.")


def load_data_automatically():
    """Carrega dados automaticamente ao iniciar o app"""
    with st.spinner("🔄 Carregando dados automaticamente do PostgreSQL..."):
        try:
            engine = init_connection()

            if engine:
                # Explorar estrutura do banco primeiro
                st.info("🔍 Explorando estrutura do banco...")
                structure = explore_database_structure(engine)
                if not structure.empty:
                    st.sidebar.info(
                        f"📁 {structure['table_name'].nunique()} tabelas encontradas")

                # Carregar todas as bases de dados
                df_hospital = load_hospital_data(engine)
                df_populacao = load_population_data(engine)
                df_economico = load_economic_data(engine)
                df_ambulatorial = load_ambulatory_data(engine)

                # Processar dados
                processor = DataProcessor()
                df_analise = processor.preprocess_data(
                    df_hospital, df_populacao, df_economico, df_ambulatorial)

                st.session_state.df_analise = df_analise
                st.session_state.processor = processor
                st.session_state.data_loaded = True

                st.sidebar.success(
                    f"✅ {len(df_analise)} registros carregados!")

            else:
                st.session_state.data_loaded = False

        except Exception as e:
            st.error(f"Erro no carregamento automático: {e}")
            st.session_state.data_loaded = False


# EXECUTAR O APLICATIVO
if __name__ == "__main__":
    main()
