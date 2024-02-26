# Importation des bibliothèques nécessaires à l'analyse de sentiments et à la visualisation

# Pandas pour la manipulation des données
import pandas as pd

# NumPy pour les opérations numériques
import numpy as np

# Matplotlib et Seaborn pour la visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK (Natural Language Toolkit) pour le traitement du langage naturel
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# TQDM pour afficher une barre de progression pendant les boucles
from tqdm.notebook import tqdm

# Hugging Face Transformers pour utiliser le modèle RoBERTa pré-entraîné
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# SciPy pour les opérations scientifiques, notamment la fonction softmax
from scipy.special import softmax


# Appliquer le style de tracé "ggplot"
plt.style.use('ggplot')

# Charger le jeu de données
df = pd.read_csv('DATA/Reviews.csv')
print(df.shape)

# Sélectionner les 500 premières lignes pour l'analyse
df = df.head(500)
print(df.shape)

# Afficher les premières lignes du jeu de données
df.head()

# Visualiser la distribution des scores de critiques
ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Nombre d\'avis par étoiles',
          figsize=(10, 5))
ax.set_xlabel('Étoiles de l\'avis')
plt.show()

# Extraire et afficher un exemple de critique
example = df['Text'][50]
print(example)

# Tokeniser l'exemple de critique en utilisant NLTK
tokens = nltk.word_tokenize(example)
tokens[:10]

# Effectuer une étiquetage des parties du discours sur les jetons
tagged = nltk.pos_tag(tokens)
tagged[:10]

# Effectuer un regroupement d'entités nommées
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

# Initialiser l'analyseur d'intensité de sentiment de NLTK
sia = SentimentIntensityAnalyzer()

# Calculer les scores de polarité à l'aide de VADER pour les critiques d'exemple
sia.polarity_scores('I am so happy!')
sia.polarity_scores('This is the worst thing ever.')
sia.polarity_scores(example)

# Exécuter le calcul des scores de polarité sur l'ensemble du jeu de données à l'aide de VADER
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

# Créer un DataFrame pour les scores de sentiment VADER
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

# Afficher les scores de sentiment et les métadonnées
vaders.head()

# Tracer les scores composés en fonction des avis étoilés Amazon à l'aide de Seaborn
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Score composé par avis étoilé Amazon')
plt.show()

# Tracer les scores positifs, neutres et négatifs par avis étoilé Amazon
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positif')
axs[1].set_title('Neutre')
axs[2].set_title('Négatif')
plt.tight_layout()
plt.show()

# Charger le modèle et le tokenizer RoBERTa pré-entraînés
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Afficher les résultats de VADER sur l'exemple de critique
print(example)
sia.polarity_scores(example)

# Encoder l'exemple de critique en utilisant le tokenizer RoBERTa et obtenir les scores de sentiment
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)

# Définir une fonction pour calculer les scores de sentiment RoBERTa pour un texte donné
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

# Exécuter les analyses de sentiment VADER et RoBERTa sur l'ensemble du jeu de données
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Erreur pour l\'ID {myid}')

# Créer un DataFrame pour les scores de sentiment combinés et les métadonnées
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

# Afficher les noms de colonnes du DataFrame combiné
results_df.columns

# Visualiser des graphiques en paire pour les scores de sentiment VADER et RoBERTa par rapport aux avis étoilés Amazon
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos'],
             hue='Score',
             palette='tab10')
plt.show()

# Afficher des exemples de critiques avec les scores les plus positifs et négatifs à la fois pour VADER et RoBERTa
results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]
results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]
results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]
results_df.query('Score == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0]
