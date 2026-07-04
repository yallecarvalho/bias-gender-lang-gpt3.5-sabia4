import streamlit as st # type: ignore
from streamlit_option_menu import option_menu # type: ignore
from utils import *

# Configuração da página inicial
st.set_page_config(page_title="Bias Analysis in LLMs", layout="wide")

# Carregar os dados
data_path = '../data/resultados_finais.csv' 
data_loader = DataLoader(data_path)
df_result = data_loader.load_data()

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 295px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Menu lateral com ícones
with st.sidebar:
    option = option_menu(
        "Menu",
        ["Home", "Data Overview",
         "Gender Bias Analysis", "Firewall Analysis", "Language Analysis"],
        icons=["house", "table", "bar-chart-line", "gender-trans", "shield-lock", "translate"],
        menu_icon="cast",
        default_index=0,
    )

bias_analysis = BiasAnalysis(data_loader)

##############################################################################################
# HOME
##############################################################################################
if option == "Home":
    st.title("Home")
    # st.write("""
    # ### Project Objective
    # To verify if a language model (LLM) has biases when identifying sentiments in sentences with different grammatical genders (masculine, feminine, and neutral).
    
    # ### Methodology
    # - Creation of various sentences, each with three versions: masculine, feminine, and neutral.
    # - Analysis of the sentences using an API of a language model (LLM).
    # - Sentiment classification in three different scales (1-3, 1-5, 1-7).
    # - Analyses performed in Portuguese and English, with and without a firewall.
    # """)

##############################################################################################
# DATA OVERVIEW
##############################################################################################
elif option == "Data Overview":
    st.title("Data Overview")
    st.dataframe(df_result)

    # data_overview = DataOverview(data_loader)

    # # Polarity analysis
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     center_text('Boxplot', 'h6')
    #     data_overview.boxplot_pol()
    # with col2:
    #     center_text('Histogram', 'h6')
    #     data_overview.histogram_pol()
    # with col3:
    #     center_text('Pie Plot', 'h6')
    #     data_overview.pieplot_pol()

    # Results
    st.header('Results')
    scale = '1-5'
    model = st.selectbox("Select Model", ['sabia-4', 'gpt-3.5-turbo'])


    plotter = Plotter(data_loader)

    col1, col2 = st.columns(2)
    with col1:
        center_text("English", 'h2')
    with col2:
        center_text("Portuguese", 'h2')

    # Boxplots
    center_text('Boxplots', 'h3')
    col1, col2 = st.columns(2)
    with col1:
        center_text('Original', 'h4')
        plotter.plot_boxplots(scale, model, 'en', 'original')
        center_text('No firewall', 'h4')
        plotter.plot_boxplots(scale, model, 'en', 'nofirewall')
    with col2:
        center_text('Original', 'h4')
        plotter.plot_boxplots(scale, model, 'pt', 'original')
        center_text('No firewall', 'h4')
        plotter.plot_boxplots(scale, model, 'pt', 'nofirewall')

    # Histograms
    center_text('Histograms', 'h3')
    col1, col2 = st.columns(2)
    with col1:
        center_text('Original', 'h4')
        plotter.plot_histograms(scale, model, 'en', 'original')
        center_text('No firewall', 'h4')
        plotter.plot_histograms(scale, model, 'en', 'nofirewall')
    with col2:
        center_text('Original', 'h4')
        plotter.plot_histograms(scale, model, 'pt', 'original')
        center_text('No firewall', 'h4')
        plotter.plot_histograms(scale, model,'pt', 'nofirewall')

    # Pie graphs
    center_text('Pie Graphs', 'h3')
    col1, col2 = st.columns(2)
    with col1:
        center_text('Original', 'h4')
        plotter.plot_pieplot(scale, model, 'en', 'original')
        center_text('No firewall', 'h4')
        plotter.plot_pieplot(scale, model, 'en', 'nofirewall')
    with col2:
        center_text('Original', 'h4')
        plotter.plot_pieplot(scale, model, 'pt', 'original')
        center_text('No firewall', 'h4')
        plotter.plot_pieplot(scale, model, 'pt', 'nofirewall')

    # Classification Reports
    center_text('Classification Reports', 'h3')
    col1, col2 = st.columns(2)
    with col1:
        center_text('Original', 'h4')
        plotter.show_classification_report(scale, model, 'en', 'original')
        center_text('No firewall', 'h4')
        plotter.show_classification_report(scale, model, 'en', 'nofirewall')
    with col2:
        center_text('Original', 'h4')
        plotter.show_classification_report(scale, model, 'pt', 'original')
        center_text('No firewall', 'h4')
        plotter.show_classification_report(scale, model, 'pt', 'nofirewall')

    # Wilcoxon test
    center_text('Wilcoxon test', 'h3')
    col1, col2 = st.columns(2)
    with col1:
        center_text('Original', 'h4')
        plotter.show_wilcoxon_test(scale, model, 'en', 'original')
        center_text('No firewall', 'h4')
        plotter.show_wilcoxon_test(scale, model, 'en', 'nofirewall')
    with col2:
        center_text('Original', 'h4')
        plotter.show_wilcoxon_test(scale, model, 'pt', 'original')
        center_text('No firewall', 'h4')
        plotter.show_wilcoxon_test(scale, model, 'pt', 'nofirewall')


##############################################################################################
# SENTIMENT ANALYSIS
##############################################################################################
# elif option == "Sentiment Analysis":
#     st.title("Sentiment Analysis")

#     bias_analysis.aggregate_differences(tipo='sentiment')
#     # Para plotar heatmap das diferenças de gênero
#     bias_analysis.plot_heatmap('diff_pol_mas', 'sentiment')
#     bias_analysis.plot_heatmap('diff_pol_fem', 'sentiment')
#     bias_analysis.plot_heatmap('diff_pol_neu', 'sentiment')


##############################################################################################
# GENDER BIAS
##############################################################################################
elif option == "Gender Bias Analysis":
    st.title("Gender Bias Analysis")
    # scale = st.selectbox("Select Scale", ['1-3', '1-5', '1-7'])
    bias_analysis.aggregate_percentage_changes()
    # bias_analysis.aggregate_differences()
    # Para plotar heatmap das diferenças de gênero
    # bias_analysis.plot_heatmap('diff_mas_fem')
    # bias_analysis.plot_heatmap('diff_mas_neu')
    # bias_analysis.plot_heatmap('diff_fem_neu')
    bias_analysis.plot_heatmap_bias('perc_change_mas_fem')
    bias_analysis.plot_heatmap_bias('perc_change_mas_neu')
    bias_analysis.plot_heatmap_bias('perc_change_fem_neu')
    st.title('')

    # bias_analysis.plot_heatmap_by_scale('1-5')
    st.title('')
    # bias_analysis.plot_heatmap_compacted_scale('1-3')
    # center_text('1-5')
    # bias_analysis.plot_heatmap_compacted_scale('1-5')
    # bias_analysis.plot_heatmap_compacted_scale('1-7')

    


##############################################################################################
# FIREWALL IMPACT
##############################################################################################
elif option == "Firewall Analysis":
    st.title("Firewall Analysis")
    bias_analysis.aggregate_firewall_percentage_changes()
    # Para plotar heatmap das diferenças de firewall (original vs nofirewall)
    bias_analysis.plot_firewall_heatmap('perc_change_masculino')
    bias_analysis.plot_firewall_heatmap('perc_change_feminino')
    bias_analysis.plot_firewall_heatmap('perc_change_neutro')
    # bias_analysis.plot_firewall_heatmap_scale('1-3')
    center_text('1-5')
    # bias_analysis.plot_firewall_heatmap_scale('1-5')
    # bias_analysis.plot_firewall_heatmap_scale('1-7')

##############################################################################################
# LANGUAGE
##############################################################################################
elif option == "Language Analysis":
    st.title("Language Analysis")
    # scale = st.selectbox("Select Scale", ['1-3', '1-5', '1-7'])
    # bias_analysis.aggregate_language_differences()
    bias_analysis.aggregate_language_percentage_changes()
    # Para plotar heatmap das diferenças de idioma (inglês vs português)
    # bias_analysis.plot_language_heatmap('diff_masculino')
    # bias_analysis.plot_language_heatmap('diff_feminino')
    # bias_analysis.plot_language_heatmap('diff_neutro')

    bias_analysis.plot_language_heatmap('perc_change_masculino')
    bias_analysis.plot_language_heatmap('perc_change_feminino')
    bias_analysis.plot_language_heatmap('perc_change_neutro')


    # bias_analysis.plot_language_heatmap_scale('1-3')
    center_text('1-5')
    # bias_analysis.plot_language_heatmap_scale('1-5')
    # bias_analysis.plot_language_heatmap_scale('1-7')
