import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st # type: ignore
import base64
from io import BytesIO
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import classification_report
from scipy.stats import wilcoxon
import numpy as np


def center_text(text, h='h6'):
    st.write(f"<{h} style='text-align: center;'>{text}</{h}>", unsafe_allow_html=True)

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.columns_results = ['resultado masculino', 'resultado feminino', 'resultado neutro']

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        return self.df

    def filter_data(self, scale, language=None, prompt=None, gender=None):
        df_filtered = self.df[self.df['escala'] == scale]

        if language:
            df_filtered = df_filtered[df_filtered['idioma'] == language]

        if prompt:
            df_filtered = df_filtered[df_filtered['prompt'] == prompt]

        return df_filtered
    
class Theme:
    def __init__(self):
        self.set_custom_theme()

    def set_custom_theme(self):
        sns.set_theme(style="whitegrid", palette="tab10")

class BasePlot:
    def __init__(self):
        self.theme = Theme()

    def plot_fig(self, fig, width, center=True):
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        img_tag = f'<img src="data:image/png;base64,{base64.b64encode(image_png).decode("utf-8")}" style="width:{width}px;"/>'
        if center:
            st.markdown(f'<div style="text-align: center;">{img_tag}</div>', unsafe_allow_html=True)
        else:
            st.markdown(img_tag, unsafe_allow_html=True)

class DataOverview(BasePlot):
    def __init__(self, data_loader):
        super().__init__()
        self.data_loader = data_loader
        self.df = data_loader.load_data()

    def boxplot_pol(self, width=400):
        fig, ax = plt.subplots()
        sns.boxplot(data=self.df['polaridade'], ax=ax)
        self.plot_fig(fig, width)

    def histogram_pol(self, width=400):
        fig, ax = plt.subplots()
        ax.hist(self.df['polaridade'], label="Polarity")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.plot_fig(fig, width)

    def pieplot_pol(self, width=400):
        fig, ax = plt.subplots()
        counts = self.df['polaridade'].value_counts().sort_index()
        counts.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90, title='Polarity', fontsize=12)
        self.plot_fig(fig, width)

class Plotter(BasePlot):
    def __init__(self, data_loader):
        super().__init__()
        self.data_loader = data_loader
        self.columns_results = ['resultado masculino', 'resultado feminino', 'resultado neutro']

    def plot_boxplots(self, scale, language, prompt, width=500):
        filtered_df = self.data_loader.filter_data(scale, language, prompt)
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_df[self.columns_results], ax=ax)
        self.plot_fig(fig, width)

    def plot_histograms(self, scale, language, prompt, width=500):
        filtered_df = self.data_loader.filter_data(scale, language, prompt)
        fig, ax = plt.subplots()
        data = [filtered_df[column] for column in self.columns_results]
        ax.hist(data, label=self.columns_results)
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        self.plot_fig(fig, width)

    def show_value_counts(self, counts):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(counts[0])
        with col2:
            st.write(counts[1])
        with col3:
            st.write(counts[2])

    def plot_pieplot(self, scale, language, prompt, width=800):
        filtered_df = self.data_loader.filter_data(scale, language, prompt)
        counts = [filtered_df[column].value_counts().sort_index() for column in self.columns_results]
        self.show_value_counts(counts)

        fig, axes = plt.subplots(1, 3)
        for i, count in enumerate(counts):
            count.plot.pie(ax=axes[i], autopct='%1.1f%%', startangle=90, title=self.columns_results[i].capitalize(), fontsize=7)
        self.plot_fig(fig, width)

    def map_results_values(self, df, scale):
        mapping = {
            '1-5': {1: 1, 2: 1, 3: 2, 4: 3, 5: 3},
            '1-7': {1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 3}
        }
        if scale == '1-3':
            return df

        df_aux = df.copy()
        for column in self.columns_results:
            df_aux[column] = df_aux[column].map(mapping.get(scale, {}))
        return df_aux

    def show_classification_report(self, scale, language, prompt):
        filtered_df = self.data_loader.filter_data(scale, language, prompt)
        filtered_df = self.map_results_values(filtered_df, scale)
        for column in self.columns_results:
            center_text(column, 'h5')
            report = classification_report(filtered_df['polaridade'], filtered_df[column], zero_division=0, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report)

    def show_wilcoxon_test(self, scale, language, prompt):
        filtered_df = self.data_loader.filter_data(scale, language, prompt)
        pairs = [('resultado masculino', 'resultado feminino'), ('resultado masculino', 'resultado neutro'), ('resultado feminino', 'resultado neutro')]
        results = [(wilcoxon(filtered_df[pair[0]], filtered_df[pair[1]])) for pair in pairs]
        w_stats = [result[0] for result in results]
        p_values = [f"{result[1]:.2e}" for result in results]
        indexes = ['Male x Female', 'Male vs Neutral', 'Female vs Neutral']
        df_wilcoxon = pd.DataFrame(data={'Statistic': w_stats, 'p-value': p_values}, index=indexes)
        st.dataframe(df_wilcoxon)

class BiasAnalysis(BasePlot):
    def __init__(self, data_loader):
        super().__init__()
        self.data_loader = data_loader
        self.columns_results = ['resultado masculino', 'resultado feminino', 'resultado neutro']

    def normalize_sentiments(self, df, column, scale):
        df_aux = df.copy()
        df_aux[column] = df_aux[column].astype(float)
        
        if scale == '1-3':
            df_aux.loc[df_aux['escala'] == scale, column] = (df_aux.loc[df_aux['escala'] == scale, column] - 1) / 2
        elif scale == '1-5':
            df_aux.loc[df_aux['escala'] == scale, column] = (df_aux.loc[df_aux['escala'] == scale, column] - 1) / 4
        elif scale == '1-7':
            df_aux.loc[df_aux['escala'] == scale, column] = (df_aux.loc[df_aux['escala'] == scale, column] - 1) / 6
        
        return df_aux

    def calculate_differences(self, df, tipo):
        for scale in ['1-3', '1-5', '1-7']:
            for column in self.columns_results:
                df = self.normalize_sentiments(df, column, scale)
        df = self.normalize_sentiments(df, 'polaridade', '1-3')

        if tipo == 'gender':
            df['diff_mas_fem'] = df['resultado masculino'] - df['resultado feminino']
            df['diff_mas_neu'] = df['resultado masculino'] - df['resultado neutro']
            df['diff_fem_neu'] = df['resultado feminino'] - df['resultado neutro']
        elif tipo == 'sentiment':
            df['diff_pol_mas'] = df['polaridade'] - df['resultado masculino']
            df['diff_pol_fem'] = df['polaridade'] - df['resultado feminino']
            df['diff_pol_neu'] = df['polaridade'] - df['resultado neutro']

        return df

    def aggregate_differences(self, tipo='gender', show=True):
        df = self.data_loader.load_data()
        df_differences = self.calculate_differences(df, tipo)
        if tipo == 'gender':
            aggregated = df_differences.groupby(['escala', 'idioma', 'prompt'])[['diff_mas_fem', 'diff_mas_neu', 'diff_fem_neu']].mean().reset_index()
        elif tipo == 'sentiment':
            aggregated = df_differences.groupby(['escala', 'idioma', 'prompt'])[['diff_pol_mas', 'diff_pol_fem', 'diff_pol_neu']].mean().reset_index()
        if show:
            st.dataframe(aggregated)
        else:
            return aggregated
        
    def aggregate_percentage_changes(self, tipo='gender', show=True):
        df = self.data_loader.load_data()

        for scale in ['1-3', '1-5', '1-7']:
            for column in self.columns_results:
                df = self.normalize_sentiments(df, column, scale)
        df = self.normalize_sentiments(df, 'polaridade', '1-3')

        if tipo == 'gender':
            # Calcula as médias para cada grupo
            aggregated_means = df.groupby(['escala', 'idioma', 'prompt'])[['resultado masculino', 'resultado feminino', 'resultado neutro']].mean().reset_index()

            # Calcula as porcentagens de mudança a partir das médias
            aggregated_means['perc_change_mas_fem'] = ((aggregated_means['resultado feminino'] - aggregated_means['resultado masculino']) / aggregated_means['resultado masculino'] * 100)
            aggregated_means['perc_change_mas_neu'] = ((aggregated_means['resultado neutro'] - aggregated_means['resultado masculino']) / aggregated_means['resultado masculino'] * 100)
            aggregated_means['perc_change_fem_neu'] = ((aggregated_means['resultado neutro'] - aggregated_means['resultado feminino']) / aggregated_means['resultado feminino'] * 100)
            
        elif tipo == 'sentiment':
            # Calcula as médias para cada grupo
            aggregated_means = df.groupby(['escala', 'idioma', 'prompt'])[['polaridade', 'resultado masculino', 'resultado feminino', 'resultado neutro']].mean().reset_index()

            # Calcula as porcentagens de mudança a partir das médias
            aggregated_means['perc_change_pol_mas'] = ((aggregated_means['resultado masculino'] - aggregated_means['polaridade']) / aggregated_means['polaridade'] * 100)
            aggregated_means['perc_change_pol_fem'] = ((aggregated_means['resultado feminino'] - aggregated_means['polaridade']) / aggregated_means['polaridade'] * 100)
            aggregated_means['perc_change_pol_neu'] = ((aggregated_means['resultado neutro'] - aggregated_means['polaridade']) / aggregated_means['polaridade'] * 100)

        if show:
            st.dataframe(aggregated_means)
        else:
            return aggregated_means


    def calculate_language_differences(self, df):
        for scale in ['1-3', '1-5', '1-7']:
            for column in ['resultado masculino', 'resultado feminino', 'resultado neutro']:
                df = self.normalize_sentiments(df, column, scale)
        df = self.normalize_sentiments(df, 'polaridade', '1-3')
        
        df_lang_differences = pd.DataFrame()
        prompts = df['prompt'].unique()
        for prompt in prompts:
            df_en = df[(df['idioma'] == 'en') & (df['prompt'] == prompt)]
            df_pt = df[(df['idioma'] == 'pt') & (df['prompt'] == prompt)]
            for scale in ['1-3', '1-5', '1-7']:
                df_en_scale = df_en[df_en['escala'] == scale]
                df_pt_scale = df_pt[df_pt['escala'] == scale]
                if not df_en_scale.empty and not df_pt_scale.empty:
                    temp_dict = {}
                    for col in ['resultado masculino', 'resultado feminino', 'resultado neutro']:
                        diff_col = f'diff_{col.split()[-1]}'
                        diff = df_en_scale[col].values - df_pt_scale[col].values
                        temp_dict[diff_col] = diff

                    temp_dict['escala'] = scale
                    temp_dict['prompt'] = prompt
                    temp_df = pd.DataFrame(temp_dict)
                    df_lang_differences = pd.concat([df_lang_differences, temp_df], ignore_index=True)
        return df_lang_differences

    def aggregate_language_differences(self, show=True):
        df = self.data_loader.load_data()
        df_lang_differences = self.calculate_language_differences(df)
        aggregated = df_lang_differences.groupby(['escala', 'prompt']).mean().reset_index()
        if show:
            st.dataframe(aggregated)
        else:
            return aggregated
        
    def aggregate_language_percentage_changes(self, show=True):
        df = self.data_loader.load_data()

        # Normaliza os sentimentos
        for scale in ['1-3', '1-5', '1-7']:
            for column in ['resultado masculino', 'resultado feminino', 'resultado neutro']:
                df = self.normalize_sentiments(df, column, scale)
        df = self.normalize_sentiments(df, 'polaridade', '1-3')

        # Calcula as médias para cada grupo (idioma, escala, prompt)
        aggregated_en = df[df['idioma'] == 'en'].groupby(['escala', 'prompt'])[['resultado masculino', 'resultado feminino', 'resultado neutro']].mean().reset_index()
        aggregated_pt = df[df['idioma'] == 'pt'].groupby(['escala', 'prompt'])[['resultado masculino', 'resultado feminino', 'resultado neutro']].mean().reset_index()

        # Une as tabelas de inglês e português
        aggregated = pd.merge(aggregated_en, aggregated_pt, on=['escala', 'prompt'], suffixes=('_en', '_pt'))

        # Calcula as variações percentuais
        for col in ['resultado masculino', 'resultado feminino', 'resultado neutro']:
            en_col = f'{col}_en'
            pt_col = f'{col}_pt'
            perc_change_col = f'perc_change_{col.split()[-1]}'
            aggregated[perc_change_col] = ((aggregated[pt_col] - aggregated[en_col]) / ((aggregated[en_col] + aggregated[pt_col]) / 2) * 100)

        if show:
            st.dataframe(aggregated)
        else:
            return aggregated
        
    def aggregate_firewall_percentage_changes(self, show=True):
        df = self.data_loader.load_data()

        # Normaliza os sentimentos
        for scale in ['1-3', '1-5', '1-7']:
            for column in ['resultado masculino', 'resultado feminino', 'resultado neutro']:
                df = self.normalize_sentiments(df, column, scale)
        df = self.normalize_sentiments(df, 'polaridade', '1-3')

        # Calcula as médias para cada grupo (idioma, escala, prompt)
        aggregated_original = df[df['prompt'] == 'original'].groupby(['escala', 'idioma'])[['resultado masculino', 'resultado feminino', 'resultado neutro']].mean().reset_index()
        aggregated_nofirewall = df[df['prompt'] == 'nofirewall'].groupby(['escala', 'idioma'])[['resultado masculino', 'resultado feminino', 'resultado neutro']].mean().reset_index()

        # Une as tabelas de inglês e português
        aggregated = pd.merge(aggregated_original, aggregated_nofirewall, on=['escala', 'idioma'], suffixes=('_original', '_nofirewall'))

        # Calcula as variações percentuais
        for col in ['resultado masculino', 'resultado feminino', 'resultado neutro']:
            en_col = f'{col}_original'
            pt_col = f'{col}_nofirewall'
            perc_change_col = f'perc_change_{col.split()[-1]}'
            aggregated[perc_change_col] = ((aggregated[pt_col] - aggregated[en_col]) / ((aggregated[en_col] + aggregated[pt_col]) / 2) * 100)

        if show:
            st.dataframe(aggregated)
        else:
            return aggregated


    def calculate_firewall_differences(self, df):
        for scale in ['1-3', '1-5', '1-7']:
            for column in ['resultado masculino', 'resultado feminino', 'resultado neutro']:
                df = self.normalize_sentiments(df, column, scale)
        df = self.normalize_sentiments(df, 'polaridade', '1-3')
        
        df_firewall_differences = pd.DataFrame()
        idioma = df['idioma'].unique()
        for idioma in idioma:
            df_original = df[(df['prompt'] == 'original') & (df['idioma'] == idioma)]
            df_nofirewall = df[(df['prompt'] == 'nofirewall') & (df['idioma'] == idioma)]
            for scale in ['1-3', '1-5', '1-7']:
                df_original_scale = df_original[df_original['escala'] == scale]
                df_nofirewall_scale = df_nofirewall[df_nofirewall['escala'] == scale]
                if not df_original_scale.empty and not df_nofirewall_scale.empty:
                    temp_dict = {}
                    for col in self.columns_results:
                        diff_col = f'diff_{col.split()[-1]}'
                        diff = df_original_scale[col].values - df_nofirewall_scale[col].values
                        temp_dict[diff_col] = diff
                    temp_dict['escala'] = scale
                    temp_dict['idioma'] = idioma
                    temp_df = pd.DataFrame(temp_dict)
                    df_firewall_differences = pd.concat([df_firewall_differences, temp_df], ignore_index=True)
        return df_firewall_differences

    def aggregate_firewall_differences(self, show=True):
        df = self.data_loader.load_data()
        df_firewall_differences = self.calculate_firewall_differences(df)
        aggregated = df_firewall_differences.groupby(['escala', 'idioma']).mean().reset_index()
        if show:
            st.dataframe(aggregated)
        else:
            return aggregated

    def plot_heatmap(self, column, tipo='gender', width=600):
        # df = self.aggregate_differences(tipo, show=False)
        df = self.aggregate_percentage_changes(tipo, show=False)
        df_pivot = df.pivot_table(values=column, index='escala', columns=['idioma', 'prompt'], aggfunc='mean')
        fig, ax = plt.subplots()
        sns.heatmap(df_pivot, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f'Diferença Média {column}')
        self.plot_fig(fig, width)

    def plot_language_heatmap(self, column, width=600):
        # df = self.aggregate_language_differences(show=False)
        df = self.aggregate_language_percentage_changes(show=False)
        df_pivot = df.pivot_table(values=column, index='escala', columns='prompt', aggfunc='mean')
        fig, ax = plt.subplots()
        sns.heatmap(df_pivot, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f'Diferença Média {column} entre Inglês e Português')
        self.plot_fig(fig, width)

    def plot_firewall_heatmap(self, column, width=600):
        # df = self.aggregate_firewall_differences(show=False)
        df = self.aggregate_firewall_percentage_changes(show=False)
        df_pivot = df.pivot_table(values=column, index='escala', columns='idioma', aggfunc='mean')
        fig, ax = plt.subplots()
        sns.heatmap(df_pivot, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f'Diferença Média {column} entre Original e No Firewall')
        self.plot_fig(fig, width)

    def plot_heatmap_by_scale(self, escala, width=600):
        # Filtra os dados pela escala específica
        # df = self.aggregate_differences('gender', show=False)
        df = self.aggregate_percentage_changes('gender', show=False)
        df_filtered = df[df['escala'] == escala]
        
        # Cria a tabela pivô para cada comparação
        df_pivot_mas_fem = df_filtered.pivot_table(values='perc_change_mas_fem', 
                                                index='idioma', 
                                                columns='prompt', 
                                                aggfunc='mean')
        
        df_pivot_mas_neu = df_filtered.pivot_table(values='perc_change_mas_neu', 
                                                index='idioma', 
                                                columns='prompt', 
                                                aggfunc='mean')
        
        df_pivot_fem_neu = df_filtered.pivot_table(values='perc_change_fem_neu', 
                                                index='idioma', 
                                                columns='prompt', 
                                                aggfunc='mean')
        
        # Cria os heatmaps para cada uma das comparações
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        sns.heatmap(df_pivot_mas_fem, annot=True, cmap="coolwarm", center=0, ax=axes[0])
        axes[0].set_title(f'Mean difference Masculine vs Feminine')
        
        sns.heatmap(df_pivot_mas_neu, annot=True, cmap="coolwarm", center=0, ax=axes[1])
        axes[1].set_title(f'Mean difference Masculine vs Neutral')
        
        sns.heatmap(df_pivot_fem_neu, annot=True, cmap="coolwarm", center=0, ax=axes[2])
        axes[2].set_title(f'Mean difference Feminine vs Neutral')
        
        # Ajusta o layout e exibe os gráficos
        plt.tight_layout()
        self.plot_fig(fig, width)

        
    def plot_heatmap_compacted_scale(self, escala, width=600):
        # Filtra os dados pela escala específica
        # df = self.aggregate_differences('gender', show=False)
        df = self.aggregate_percentage_changes('gender', show=False)
        df_filtered = df[df['escala'] == escala]
        
        # Cria um DataFrame consolidado com as diferenças
        df_consolidated = pd.DataFrame({
            'perc_change_mas_fem': df_filtered['perc_change_mas_fem'],
            'perc_change_mas_neu': df_filtered['perc_change_mas_neu'],
            'perc_change_fem_neu': df_filtered['perc_change_fem_neu'],
            'prompt': df_filtered['prompt'],
            'idioma': df_filtered['idioma']
        })

        # Faz a tabela pivô garantindo a ordem correta das comparações
        df_pivot = df_consolidated.melt(id_vars=['prompt', 'idioma'], 
                                        value_vars=['perc_change_mas_fem', 'perc_change_mas_neu', 'perc_change_fem_neu'],
                                        var_name='Comparação',
                                        value_name='Diferença')
        
        df_pivot['Comparação'] = pd.Categorical(df_pivot['Comparação'], 
                                                categories=['perc_change_mas_fem', 'perc_change_mas_neu', 'perc_change_fem_neu'], 
                                                ordered=True)
        
        df_pivot = df_pivot.pivot_table(values='Diferença', 
                                        index='Comparação', 
                                        columns=['idioma', 'prompt'],
                                        aggfunc='mean')
        
        # Cria o heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_pivot, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f'Diferença Média entre Gêneros (Escala {escala})')
        
        # Ajusta o layout e exibe o gráfico
        plt.tight_layout()
        self.plot_fig(fig, width)


    def plot_firewall_heatmap_scale(self, escala, width=600):
        # Filtra os dados pela escala específica
        # df = self.aggregate_firewall_differences(show=False)
        df = self.aggregate_firewall_percentage_changes(show=False)
        df_filtered = df[df['escala'] == escala]
        
        # Cria um DataFrame consolidado com as diferenças
        df_consolidated = pd.DataFrame({
            'perc_change_masculine': df_filtered['perc_change_masculino'],
            'perc_change_feminine': df_filtered['perc_change_feminino'],
            'perc_change_neutral': df_filtered['perc_change_neutro'],
            'Language': df_filtered['idioma']
        })

        # Faz a tabela pivô
        df_pivot = df_consolidated.melt(id_vars=['Language'], 
                                        value_vars=['perc_change_masculine', 'perc_change_feminine', 'perc_change_neutral'],
                                        var_name='Comparação',
                                        value_name='Diferença')
        
        df_pivot['Comparação'] = pd.Categorical(df_pivot['Comparação'], 
                                                categories=['perc_change_masculine', 'perc_change_feminine', 'perc_change_neutral'], 
                                                ordered=True)
        
        # Certifique-se de que observed=False está incluído em todas as chamadas
        df_pivot = df_pivot.pivot_table(values='Diferença', 
                                        index='Comparação', 
                                        columns='Language',
                                        aggfunc='mean',
                                        observed=False)  # Especifica observed=False para silenciar o aviso
        
        # Cria o heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_pivot, annot=True, cmap="coolwarm", center=0, ax=ax)
        # ax.set_title(f'Diferença Média por Gênero e Idioma (Escala {escala})')
        ax.set_ylabel('')
        ax.set_xlabel('Language')
        
        # Ajusta o layout e exibe o gráfico
        plt.tight_layout()
        self.plot_fig(fig, width)


    def plot_language_heatmap_scale(self, escala, width=600):
        # Filtra os dados pela escala específica
        # df = self.aggregate_language_differences(show=False)
        df = self.aggregate_language_percentage_changes(show=False)
        df_filtered = df[df['escala'] == escala]
        
        # Cria um DataFrame consolidado com as diferenças
        df_consolidated = pd.DataFrame({
            'perc_change_masculine': df_filtered['perc_change_masculino'],
            'perc_change_feminine': df_filtered['perc_change_feminino'],
            'perc_change_neutral': df_filtered['perc_change_neutro'],
            'Prompt': df_filtered['prompt']
        })

        # Faz a tabela pivô
        df_pivot = df_consolidated.melt(id_vars=['Prompt'], 
                                        value_vars=['perc_change_masculine', 'perc_change_feminine', 'perc_change_neutral'],
                                        var_name='Comparação',
                                        value_name='Diferença')
        
        df_pivot['Comparação'] = pd.Categorical(df_pivot['Comparação'], 
                                                categories=['perc_change_masculine', 'perc_change_feminine', 'perc_change_neutral'], 
                                                ordered=True)
        
        df_pivot = df_pivot.pivot_table(values='Diferença', 
                                        index='Comparação', 
                                        columns='Prompt',
                                        aggfunc='mean',
                                        observed=False)  # Especifica observed=False para silenciar o aviso
        
        # Cria o heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_pivot, annot=True, cmap="coolwarm", center=0, ax=ax)
        # ax.set_title(f'Mean difference by Language')
        ax.set_ylabel('')
        
        # Ajusta o layout e exibe o gráfico
        plt.tight_layout()
        self.plot_fig(fig, width)



def count_and_plot_en(df):
    """Print value counts and pie charts for the English regard results."""
    count_pol = df['polaridade'].value_counts().sort_index()
    count_mas = df['resultado masculino'].value_counts().sort_index()
    count_fem = df['resultado feminino'].value_counts().sort_index()
    count_neu = df['resultado neutro'].value_counts().sort_index()

    print(count_pol)
    print(count_mas)
    print(count_fem)
    print(count_neu)

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    count_pol.plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90, title='True Distribution')
    count_mas.plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90, title='Masculine Result')
    count_fem.plot.pie(ax=axes[2], autopct='%1.1f%%', startangle=90, title='Feminine Result')
    count_neu.plot.pie(ax=axes[3], autopct='%1.1f%%', startangle=90, title='Neutral Result')
    plt.tight_layout()
    plt.show()


def calculate_and_display_metrics_en(df):
    """Print a classification report for each gender column against the true polarity."""
    columns = ['resultado masculino', 'resultado feminino', 'resultado neutro']
    for col in columns:
        print(col)
        print(classification_report(df['polaridade'], df[col], zero_division=0))
        print()


def wilcoxon_test_en(df):
    """Run pairwise Wilcoxon signed-rank tests between polarity and gender result columns."""
    w_stat_pol_mas, p_val_pol_mas = wilcoxon(df['polaridade'], df['resultado masculino'])
    w_stat_pol_fem, p_val_pol_fem = wilcoxon(df['polaridade'], df['resultado feminino'])
    w_stat_pol_neu, p_val_pol_neu = wilcoxon(df['polaridade'], df['resultado neutro'])
    w_stat_mas_fem, p_val_mas_fem = wilcoxon(df['resultado masculino'], df['resultado feminino'])
    w_stat_mas_neu, p_val_mas_neu = wilcoxon(df['resultado masculino'], df['resultado neutro'])
    w_stat_fem_neu, p_val_fem_neu = wilcoxon(df['resultado feminino'], df['resultado neutro'])
    return {
        'pol vs mas': (w_stat_pol_mas, p_val_pol_mas),
        'pol vs fem': (w_stat_pol_fem, p_val_pol_fem),
        'pol vs neu': (w_stat_pol_neu, p_val_pol_neu),
        'mas vs fem': (w_stat_mas_fem, p_val_mas_fem),
        'mas vs neu': (w_stat_mas_neu, p_val_mas_neu),
        'fem vs neu': (w_stat_fem_neu, p_val_fem_neu),
    }


def save_results_en(df, scale, resultado_mas, resultado_fem, resultado_neu, prompt=''):
    """Save English regard results to a CSV file.

    Mirrors the behaviour of Prompt.save_results() but as a standalone function
    for use in the exploratory notebooks (gpt-pol5, gpt-pol7) that call it
    outside the Prompt class.

    Parameters
    ----------
    df : pd.DataFrame
        The base dataframe (frases-generos.csv or similar).
    scale : int or str
        Scale used (e.g. 3, 5 or 7).
    resultado_mas : list
        Regard scores for the masculine sentences.
    resultado_fem : list
        Regard scores for the feminine sentences.
    resultado_neu : list
        Regard scores for the neutral sentences.
    prompt : str, optional
        Prompt variant identifier (e.g. '' for original or 'nofirewall').
    """
    df_out = df.copy()
    df_out['resultado masculino'] = resultado_mas
    df_out['resultado feminino'] = resultado_fem
    df_out['resultado neutro'] = resultado_neu

    suffix = f'_{prompt}' if prompt else '_'
    filename = f'data/resultados_gpt_{scale}_en{suffix}.csv'
    df_out.to_csv(filename, index=False)
    print(f'Saved: {filename}')


def count_and_plot(df):
    """Print value counts and pie charts for the Portuguese regard results."""
    count_pol = df['polaridade'].value_counts().sort_index()
    count_mas = df['resultado masculino'].value_counts().sort_index()
    count_fem = df['resultado feminino'].value_counts().sort_index()
    count_neu = df['resultado neutro'].value_counts().sort_index()

    print(count_pol)
    print(count_mas)
    print(count_fem)
    print(count_neu)

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    count_pol.plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90, title='Distribuição Verdadeira')
    count_mas.plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90, title='Resultado Masculino')
    count_fem.plot.pie(ax=axes[2], autopct='%1.1f%%', startangle=90, title='Resultado Feminino')
    count_neu.plot.pie(ax=axes[3], autopct='%1.1f%%', startangle=90, title='Resultado Neutro')
    plt.tight_layout()
    plt.show()


def calculate_and_display_metrics(df):
    """Print a classification report for each gender column against the true polarity."""
    columns = ['resultado masculino', 'resultado feminino', 'resultado neutro']
    for col in columns:
        print(col)
        print(classification_report(df['polaridade'], df[col], zero_division=0))
        print()


def plot_boxplots(df):
    """Plot boxplots for the three gender result columns."""
    columns = ['resultado masculino', 'resultado feminino', 'resultado neutro']
    df_melted = df[columns].melt(var_name='Gênero', value_name='Resultado')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Gênero', y='Resultado', data=df_melted, ax=ax)
    ax.set_title('Distribuição dos resultados por Gênero')
    plt.show()


def descriptive_stats(df):
    """Return descriptive statistics for the result columns."""
    columns = ['resultado masculino', 'resultado feminino', 'resultado neutro']
    return df[columns].describe()


def wilcoxon_test(df):
    """Run pairwise Wilcoxon signed-rank tests between polarity and gender result columns."""
    w_stat_pol_mas, p_val_pol_mas = wilcoxon(df['polaridade'], df['resultado masculino'])
    w_stat_pol_fem, p_val_pol_fem = wilcoxon(df['polaridade'], df['resultado feminino'])
    w_stat_pol_neu, p_val_pol_neu = wilcoxon(df['polaridade'], df['resultado neutro'])
    w_stat_mas_fem, p_val_mas_fem = wilcoxon(df['resultado masculino'], df['resultado feminino'])
    w_stat_mas_neu, p_val_mas_neu = wilcoxon(df['resultado masculino'], df['resultado neutro'])
    w_stat_fem_neu, p_val_fem_neu = wilcoxon(df['resultado feminino'], df['resultado neutro'])
    return {
        'pol vs mas': (w_stat_pol_mas, p_val_pol_mas),
        'pol vs fem': (w_stat_pol_fem, p_val_pol_fem),
        'pol vs neu': (w_stat_pol_neu, p_val_pol_neu),
        'mas vs fem': (w_stat_mas_fem, p_val_mas_fem),
        'mas vs neu': (w_stat_mas_neu, p_val_mas_neu),
        'fem vs neu': (w_stat_fem_neu, p_val_fem_neu),
    }

def map_polarity_values(df):
    mapping_results = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
    for col in ['resultado masculino', 'resultado feminino', 'resultado neutro']:
        df[col] = df[col].map(mapping_results)
    return df