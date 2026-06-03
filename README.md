# bias-gender-lang-gpt3.5

Code and data for the paper **["Biases in GPT-3.5 Turbo model: a case study regarding gender and language"](https://aclanthology.org/2024.stil-1.4/)** (STIL 2024).

## Installation

```bash
pip install -r requirements.txt
```

## Running the Streamlit app

```bash
cd code
streamlit run app.py
```

## Running the notebooks

Open the notebooks in `notebooks/` with Jupyter. The notebooks require an OpenAI API key set as the environment variable `OPENAI_API_KEY` to re-run the prompts; the analysis notebooks can be run without it using the data already in `data/`.

## Project structure

```
├── data/
│   ├── frases.csv                  # Original English sentences from Sheng et al. (2019) with regard polarity scores
│   ├── frases-traduzidas.csv       # Automatic Portuguese translations (Google Translate API)
│   ├── frases-generos.csv          # Final dataset: 466 sentences × 3 genders × 2 languages
│   └── results_gender2.csv         # GPT-3.5-turbo regard scores for all conditions
│
├── code/
│   ├── app.py                      # Streamlit app for interactive result exploration
│   └── utils.py                    # Helper classes and functions used by the app and notebooks
│
└── notebooks/
    ├── traducao_bases.ipynb        # Translates frases.csv to Portuguese and generates frases-traduzidas.csv
    ├── divide-genero.ipynb         # Generates the gendered sentence versions and builds frases-generos.csv
    ├── run_prompts.ipynb           # Sends prompts to GPT-3.5-turbo and saves regard scores
    ├── gpt-pol3.ipynb              # Analysis of results on the 1–3 scale (Portuguese)
    ├── gpt-pol5.ipynb              # Analysis of results on the 1–5 scale (English)
    └── gpt-pol7.ipynb              # Analysis of results on the 1–7 scale (English)
```