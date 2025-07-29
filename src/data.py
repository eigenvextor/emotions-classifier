import numpy as np
import pandas as pd 

import re

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

df1 = pd.read_csv('../data/goemotions_1.csv')
df2 = pd.read_csv('../data/goemotions_2.csv')
df3 = pd.read_csv('../data/goemotions_3.csv')

df = pd.concat([df1, df2, df3]).reset_index(drop=True)
cols = ['id', 'text' ,'admiration',
       'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
       'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
       'remorse', 'sadness', 'surprise', 'neutral']

print("\nno of unclear tweets:\n", df.example_very_unclear.value_counts())

"""- text: The text of the comment (with masked tokens, as described in the paper).
- id: The unique id of the comment.
- author: The Reddit username of the comment's author.
- subreddit: The subreddit that the comment belongs to.
- link_id: The link id of the comment.
- parent_id: The parent id of the comment.
- created_utc: The timestamp of the comment.
- rater_id: The unique id of the annotator.
- example_very_unclear: Whether the annotator marked the example as being very unclear or 
                    difficult to label (in this case they did not choose any emotion labels).
"""

no_text_cols = cols.copy()
no_text_cols.remove('text')

print('\nno of examples before preprocessing:', df['id'].nunique())
print('no of unique rater ids:', df['rater_id'].nunique())
print('no of unique tweets:', df['text'].nunique())
print('unclear/ difficult to label tweets (from df):', np.round((df['example_very_unclear'].sum()*100 / len(df)), 2), '%')

aggregated = df[no_text_cols].groupby('id').sum()
raters_2 = (aggregated >= 2).any(axis=1).sum()
raters_3 = (aggregated >= 3).any(axis=1).sum()

print("no of tweets where at least 2+ raters agree upon atleast 1 label:", raters_2)
print("no of tweets where at least 3+ raters agree upon atleast 1 label:", raters_3)

prop = df.groupby('id')['rater_id'].nunique().value_counts(normalize=True)*100
print("\nno of raters per tweet (id):\n", prop)

emotion_cols = cols.copy()
emotion_cols.remove('id')
emotion_cols.remove('text')

def first_preprocessing(text):
    all_punctuations = r'''!-{},<>.\/?@$%^&*_~`|()#'''
    # >1 same punctuations replaced by same punctuation
    cleaned_text = re.sub(rf'([{re.escape(all_punctuations)}])\s*\\1*', r'\1 ', text)
    remove_punctuations = r'''{}<>\/@#$%^&*_~`|()'''
    # removing less occuring/ noisy punctuations
    cleaned_text = ''.join(char for char in cleaned_text if char not in remove_punctuations)
    # remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    # make sure there is no extra space after sentence complete
    if len(cleaned_text) != 0:
        if cleaned_text[-1] == ' ':
            return cleaned_text[:-1]

    return cleaned_text

new_df = df.copy()
new_df = new_df[new_df['example_very_unclear']==False].reset_index(drop=True)
new_df = new_df[['id', 'text']][~new_df[['id']].duplicated()]
new_df = pd.merge(df[no_text_cols].groupby('id').sum() >= 2, new_df, on='id')
new_df = new_df[new_df.drop(columns={'text', 'id'}).sum(axis=1) >= 1]

new_df = new_df[cols]
new_df['text'] = new_df['text'].apply(first_preprocessing)
new_df[emotion_cols] = new_df[emotion_cols].astype(int)

df1 = new_df[~new_df.duplicated('text', keep=False)].reset_index(drop=True).drop(columns={'id'})
df2 = new_df[new_df.duplicated('text', keep=False)].reset_index(drop=True)
df2 = (df2.drop(columns={'id'})[df2.duplicated('text', keep=False)].groupby('text').sum() >= 2).reset_index()

final_df = pd.concat([df1, df2]).reset_index(drop=True)
l_index, u_index = [], []
for i, text in enumerate(final_df['text']):
    if len((text).split()) < 3:
        l_index.append(i)
    if len((text).split()) > 30:
        u_index.append(i)

print('\nno of texts with less than 3 words:', len(l_index))
print('no of texts with more than 30 words:', len(u_index))

print('\nno of examples after preprocessing:', len(final_df))
final_df.to_csv('../data/cleaned_data.csv', index=False)
print()
