import datasets
import html
import re

def _clean(entry):
    ''' the text is scraped HTML, this cleans html noise '''
    text = html.unescape(entry['text']) # html code -> unicode
    text = re.sub(r'<[^>]+>', ' ', text) # strip tags from html
    text = re.sub(r'\s+', ' ', text).strip() # multiple spaces become one space
    return {'text': text}

def get_ds():
    ''' returns hf dataset objects from the imdb dataset '''
    dataset = datasets.load_dataset('imdb') # this is a dict of hf dataset objects
    dataset = dataset.map(_clean) # clean html noise, maps each item of the dict
    train_data = dataset['train'] # train set with 25k entries
    test_data = dataset['test'] # test set 25k entries
    unsupervised_data = dataset['unsupervised'] # pre-train set with 50k unlabled entries
    return train_data, test_data, unsupervised_data