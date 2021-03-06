from bs4 import BeautifulSoup
from requests import get
import pandas as pd
import numpy as np
from time import sleep
from typing import List, Dict, Tuple, Union
import os
import string
import spacy
from spacy.pipeline import merge_entities
import en_core_web_lg
import re


def url_to_file(df: pd.DataFrame, url_col: str, dir_: str) -> List[Tuple[int, str]]:
    """
    This function takes in a dataframe with url to texts, 
    save the texts a .txt file for each url and 
    return bad urls for hand scraping.

    Args:
        df (pd.DataFrame): dataframe that contains the url links to the texts
        url_col ([str]): name of the url column
        dir_ (str): directory where the .txt files will be saved

    Returns:
        List[Tuple[int, str]]: List of tuples containing the index and bad urls.
    """

    bad_url = []
    for id_, url in enumerate(df[url_col]):
        response = get(url)
        if response.status_code==200:
            soup = BeautifulSoup(response.text, 'lxml')
            results = soup.find_all(text=True)
            text = ''
            for result in results:
                text = text + ' ' + result
            with open(dir_+f'{id_}.txt', 'w') as file:
                file.write(text)
        else:
            bad_url.append((id_, url))
        sleep(5)
    return bad_url

def add_text_to_df(dir_: str, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    This funciton extract texts from all .txt files in a directory and
    fill them in the dataframe matching file names by indices. 

    Args:
        dir_ (str): directory of all .txt files
        df (pd.DataFrame): dataframe containing indices, source urls and classes
        col_name (str): name of the column for the text

    Returns:
        pd.DataFrame: dataframe containing, indices, source urls, classes and text copus.
    """
    for f in os.scandir(dir_):
        if f.path.split('.')[-1] == 'txt':
            idx = int(f.name.split('.')[0])
            with open(f.path, 'r') as file:
                text = file.read().replace('\n', ' ')
                df[col_name][idx] = text
    
    return df

def text_to_df(dir_: str, df: pd.DataFrame, col_names: List[str]) ->pd.DataFrame:
    """
    This function extract from all .txt files in a directory and 
    append them to the dataframe and record class labels as folder name

    Args:
        dir_ (str): directory
        df (pd.DataFrame): dataframe 
        col_name (str): list of two columns: text and class

    Returns:
        pd.DataFrame: dataframe with texts and classes appended
    """

    label = int(dir_.split('/')[-2])
    texts = []
    for f in os.scandir(dir_):
        if f.path.split('.')[-1] == 'txt':
            with open(f.path, 'r') as file:
                text = file.read().replace('\n', ' ')
                texts.append(text)
    addition = pd.DataFrame(columns=[col_names[0]], data=texts)
    addition[col_names[1]] = label
    return pd.concat([df, addition], axis=0, join='outer', ignore_index=True)

PERSON = ['PERSON']
PLACE = ['FAC', 'GPE', 'LOC']
ORG = ['NORP', 'ORG']
TIME = ['DATE', 'TIME']
NUM = ['PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

PAST = ['VBD']
PRESENT = ['VBP', 'VBZ']
MODAL = ['MD']
AD = ['ADJ', 'ADV']
PROB = ['definitely', 'certainly', 'clearly', 'obviously', 'possibly', 'perhaps', 'probably', 'maybe', 'possible', 'definite', 'certain', 'obvious', 'likely']

def tokenize_ft_extraction(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    This function takes in a dataframe and column name for text
    and performs the following to the text:
    remove punctuation;
    change to lower cases;
    tokenization, including tokenize numbers as [NUM];
    lemmatization


    Args:
        df (pd.DataFrame): dataframe to be transformed
        col_name (str): column name of the text

    Returns:
        pd.DataFrame: transformed dataframe
    """
    punctuation = string.punctuation
    df[col_name] = df[col_name].str.replace('['+punctuation+']', '', regex=True)
    df[col_name] = df[col_name].str.lower().str.strip()
    nlp = en_core_web_lg.load()
    nlp.add_pipe(merge_entities)
    df[col_name] = df[col_name].apply(_regex_clean)
    lemmatized_text = []
    df['ents_rep'] = None
    df['vocab'] = None
    df['ppo_rep'] = None
    df['no_ents_text'] = None
    df['verb_present'] = None
    df['verb_past'] = None
    df['verb'] = None
    df['modal'] = None
    df['ad'] = None
    df['prob'] = None
    for idx, text in enumerate(df[col_name]):
        doc = nlp(text)
        tokens = []
        ents = []
        ppo = []
        verb_present = []
        verb_past = []
        verb = []
        modal = []
        ad = []
        prob = []
        for token in doc:
            if token.is_stop:
                continue # drop stop words
            elif token.lemma_=='-PRON-': # replace PRON with original text
                tokens.append(token.text)
            elif not token.ent_type_:
                tokens.append(token.lemma_) # use lemmatized tokens for non NE tokens
                if token.text in PROB:
                    prob.append(token.text)
                elif token.tag_ in PAST:
                    verb_past.append(token.lemma_)
                    verb.append(token.lemma_)
                elif token.tag_ in PRESENT:
                    verb_present.append(token.lemma_)
                    verb.append(token.lemma_)
                elif token.tag_ in ['VB', 'VBG', 'VBN']:
                    verb.append(token.lemma_)
                elif token.tag_ in MODAL:
                    modal.append(token.text)
                elif token.pos_ in AD:
                    ad.append(token.text)
            else:
                tokens.append(token.ent_type_)
                ents.append(token.text.lower()) # keep NE token for calculating repeated NE
                if token.ent_type_ in [*PERSON, *PLACE, *ORG]:
                    ppo.append(token.text.lower()) # keep PPO person, place, organisation NE for calculating repeated NE
        lemmatized_text.append(tokens)
        df.loc[idx, 'ents_rep'] = len(ents)/len(set(ents)) # repeat ratio of NE
        df.loc[idx, 'vocab'] = len(set(tokens))/len(tokens) # normalised vocabulary size
        if len(ppo)!=0:
            df.loc[idx, 'ppo_rep'] = len(ppo)/(len(set(ppo))) # repeat retio of PPO
        else:
            df.loc[idx, 'ppo_rep'] = 0
        df.loc[idx, 'no_ents_text'] = ' '.join(tokens)
        df.loc[idx, 'verb_present'] = len(verb_present)/len(tokens) # normalised present tense verb count
        df.loc[idx, 'verb_past'] = len(verb_past)/len(tokens) # normalised past tense verb count
        df.loc[idx, 'verb'] = len(verb)/len(tokens) # normalised verb count
        df.loc[idx, 'modal'] = len(modal)/len(tokens) # normalised modal count
        df.loc[idx, 'ad'] = len(ad)/len(tokens) # normalised adjective/adverb count
        df.loc[idx, 'prob'] = len(prob)/len(tokens) # normalised count of advs and adjs of probability

    df['lem_text'] = lemmatized_text
    df = _feature_extraction(df, 'lem_text')
    return df

def _feature_extraction(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    This function takes in tokenized dataframe and text column name and extracts feaures.

    Args:
        df (pd.DataFrame): tokenized dataframe
        col_name (str): name of the column containing texts

    Returns:
        pd.DataFrame: tokenized datafram with hand crafted features.
    """
    df['len'] = df[col_name].apply(lambda x: len(x))
    df['org_count'] = df[col_name].apply(lambda x: sum(x.count(org)/len(x) for org in ORG))
    df['place_count'] = df[col_name].apply(lambda x: sum(x.count(place)/len(x) for place in PLACE))
    df['time_count'] = df[col_name].apply(lambda x: sum(x.count(time)/len(x) for time in TIME))
    df['person_count'] = df[col_name].apply(lambda x: x.count('PERSON')/len(x))
    df['num_count'] = df[col_name].apply(lambda x: sum(x.count(num)/len(x) for num in NUM))
    df['ne_count'] = df['org_count'] + df['person_count'] + df['place_count']

    return df

def _regex_clean(text: str) -> str:
    """
    This function strips emails and urls from a given string

    Args:
        text (str): text

    Returns:
        str: stripped text
    """
    # text = re.sub(r'\d+', '[NUM]', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    return text

def back_translation(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    This function takes in a dataframe, backtranslate the text column and append to the original dataframe with labels 

    Args:
        df (pd.DataFrame): Dataframe containing text and labels
        col_name (str): Name of the text column

    Returns:
        pd.DataFrame: Dataframe containing original text and backtranslated text and labels
    """

    MODELS = {
        'GERMAN': ('transformer.wmt19.en-de', 'transformer.wmt19.de-en'),
        'RUSSIAN': ('transformer.wmt19.en-ru', 'transformer.wmt19.ru-en')
    }

    for _, model in MODELS.items():
        back_translation_aug = naw.BackTranslationAug(
            from_model_name=model[0],
            to_model_name=model[1]
        )
        translation = df['col_name'].apply(back_translation_aug.augment)
        labels = df['class_id']
        trans_df = pd.DataFrame({
            'text': translation,
            'class_id': labels
        })

        df = pd.concat([df, trans_df], axis=0, ignore_index=True)
    
    return df