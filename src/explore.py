import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import re
import contractions
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from itertools import combinations

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('fivethirtyeight')

# # download nltk data
# path = '../venv/nltk_data'
# nltk.data.path.append(path)
# nltk.download('punkt', download_dir=path)
# nltk.download('punkt_tab', download_dir=path)
# nltk.download('stopwords', download_dir=path)
# nltk.download('wordnet', download_dir=path)

df = pd.read_csv('../data/cleaned_data.csv')

def preprocessing(text):
    text = text.lower()
    # remove emojis/ non-english words
    allowed_punctuation = ['!', '"', "'", '?', '.', ',', '[', ']']
    pattern = r'[^a-zA-Z0-9\s' + re.escape(''.join(allowed_punctuation)) + ']'
    cleaned_text = re.sub(pattern, '', text)

    # replace numbers with [NUM] tag
    cleaned_text = re.sub(r'\d+', ' [NUM] ', cleaned_text)

    # handle contractions
    cleaned_text = contractions.fix(cleaned_text)

    # Replace space before period and comma with no space
    cleaned_text = re.sub(r'\s+([.,!?\'\"])', r'\1', cleaned_text)
    # Add space after period and comma
    cleaned_text = re.sub(r'([.,!?])', r'\1 ', cleaned_text)

    # remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    tokens = word_tokenize(cleaned_text.lower())

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    stop_words.remove('no')
    cleaned_tokens = [word for word in tokens if word not in stop_words]

    cleaned_text = ' '.join(cleaned_tokens)

    # Replace [ name ] with [name]
    cleaned_text = re.sub(r'\[ *name *\]', '[NAME]', cleaned_text)
    # Replace [ religion ] with [religion]
    cleaned_text = re.sub(r'\[ *religion *\]', '[RELIGION]', cleaned_text)
    # Replace [ num ] with [num]
    cleaned_text = re.sub(r'\[ *num *\]', '[NUM]', cleaned_text)
    
    return cleaned_text

df['preprocessed_text'] = df['text'].map(lambda x: preprocessing(x))

def nltk_preprocessing(text):
    # lemmatize words
    tokens = text.lower().split()
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    cleaned_text = ' '.join(cleaned_tokens)

    # replace [ name ] with [name]
    cleaned_text = re.sub(r'\[ *name *\]', '[NAME]', cleaned_text)
    # replace [ religion ] with [religion]
    cleaned_text = re.sub(r'\[ *religion *\]', '[RELIGION]', cleaned_text)
    # replace [ num ] with [num]
    cleaned_text = re.sub(r'\[ *num *\]', '[NUM]', cleaned_text)

    # define regular expression patterns to match [NAME] and [NUM]
    name_pattern = r'\[NAME\]'
    num_pattern = r'\[NUM\]'
    reg_pattern = r'\[REG\]'

    # replace [NAME] and [NUM] occurrences with placeholders to protect them
    protected_text = re.sub(name_pattern, '1', text)
    protected_text = re.sub(num_pattern, '2', protected_text)
    protected_text = re.sub(reg_pattern, '3', protected_text)

    # remove punctuation from the text
    cleaned_text = re.sub(r'[^\w\s]', '', protected_text)

    # restore [NAME] and [NUM] occurrences
    final_text = re.sub(r'1', '[NAME]', cleaned_text)
    final_text = re.sub(r'2', '[NUM]', final_text)
    final_text = re.sub(r'3', '[REG]', final_text)

    return final_text

df['nltk_preprocessed'] = df['preprocessed_text'].map(lambda x: nltk_preprocessing(x))
dfc = df[df.columns[df.dtypes=='int64']]
dfc.head()

fig, ax = plt.subplots(figsize=(12,8))
data = dfc.sum()
sns.barplot(x=data[:-1].values, y=data[:-1].index, ax=ax, color='orange');

for i, v in enumerate(data[:-1].values):
    ax.text(v + 50, i+0.2, str(v), color='black')

ax.set_title(f'emotions (excl neutral: {data[-1]})', pad=30);
ax.set_ylabel(f'')
fig.subplots_adjust(left=0.15)
plt.savefig("../img/no-of-emotions.png", dpi=150, bbox_inches='tight')

emotions = tuple(dfc.columns)

two_combinations = []
for col1, col2 in tqdm(combinations(dfc.columns, 2), desc="two combinations"):
    label = f'{col1}-{col2}'
    count = (df[[col1, col2]].sum(axis=1) == 2).sum()
    two_combinations.append([label, count])
two_combinations = sorted(two_combinations, key=lambda x: x[1], reverse=True)
two_combinations_label = [sublist[0] for sublist in two_combinations]
two_combinations_val = [sublist[1] for sublist in two_combinations]

fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(x=two_combinations_val[:20], y=two_combinations_label[:20], ax=ax, color='orange');
for i, v in enumerate(two_combinations_val[:20]):
    ax.text(v + 5, i+0.2, str(v), color='black')
ax.set_title(f'Top 20 Two-Category Combinations', pad=30)
fig.subplots_adjust(left=0.25)
plt.savefig("../img/two-combinations.png", dpi=150, bbox_inches='tight')

three_combinations = []
for col1, col2, col3 in tqdm(combinations(dfc.columns, 3), desc="three combinations"):
    label = f'{col1}-{col2}-{col3}'
    count = (df[[col1, col2, col3]].sum(axis=1) == 3).sum()
    three_combinations.append([label, count])
three_combinations = sorted(three_combinations, key=lambda x: x[1], reverse=True)
three_combinations_label = [sublist[0] for sublist in three_combinations]
three_combinations_val = [sublist[1] for sublist in three_combinations]

fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(x=three_combinations_val[:20], y=three_combinations_label[:20], ax=ax, color='orange');
for i, v in enumerate(three_combinations_val[:20]):
    ax.text(v + 0.2, i+0.2, str(v), color='black')
ax.set_title(f'Top 20 Three-Category Combinations', pad=30)
fig.subplots_adjust(left=0.3)
plt.tight_layout()
plt.savefig("../img/three-combinations.png", dpi=150, bbox_inches='tight')
# plt.show()