import datasets
import tokenizer
from dataset import get_ds

# DATASET

if __name__ == '__main__':

    # get dataset
    train_data, test_data, unsupervised_data = get_ds()

    # load or train and load tokenizer
    tok = tokenizer.load_tokenizer()
    if tok is None:
        data = datasets.concatenate_datasets([train_data, test_data, unsupervised_data])
        tok = tokenizer.train_tokenizer(list(data['text']))

    # test tokenizer
    text = train_data[0]['text'][:50]
    print(f'String: {text}')
    encoded = tokenizer.encode(tok, text)
    print(f'Encoded: {encoded}')
    print(f'Decode: {tokenizer.decode(tok, encoded)}')


