import os
from tokenizers import Tokenizer

def train_tokenizer(text_list):
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
    from tokenizers.processors import TemplateProcessing

    # init tokenizer object, we use Byte-Pair Encoding
    tokenizer = Tokenizer(BPE(unk_token='[unknown]')) # unknown token will be placed when unknown characters are found

    # preprocessing step on input string
    #   - Some unicode charactes can be represented in two ways, 'é' and ('e', '´') are two representations of the same thing
    #     NFD will guarantee every character has only one representation in unicode
    #   - Lowercase() turns everything into lowercase
    #   - StripAccents() removes accents post NFD normalization
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

    # splits string whenever it finds punctuation ('!', '.', ...), keeps tokenizer from merging sequences that span 2 different words or word + punctuation
    tokenizer.pre_tokenizer = Whitespace()

    # trainer object holds parameters for tokenization
    trainer = BpeTrainer(
        vocab_size=10000, # unique tokens in final vocab !!! SHOULD BE CONFIGURABLE
        special_tokens=['[unknown]', '[padding]', '[classification]', '[separator]', '[mask]'], # tokens for nlp
        min_frequency=2, # a token must follow the other at least two times for a merge to happen
    )

    # for reference on the special tokens above:
    # '[unknown]' is used for unknown characters that the tokenizer might find in deployment (saves us from crashes)
    # '[padding]' is used to handle different sequence lenghts in our NLP model
    # '[classification]' is the token that mapps to the output classification on our network, it sits before the input, 
    # the newtork learns to encode the information necessary for pred there
    # '[separator]' is a token that sits in the end of the input, tells network the review has ended
    # '[mask]' is used for pretraining the model on reviews that have no label

    # train the tokenizer on the raw strings
    tokenizer.train_from_iterator(text_list, trainer=trainer, length=len(text_list))

    # configure the tokenizer so the ['classification'] token is appended at the start and
    # the ['separator'] at the end of the input on "inference"
    tokenizer.post_processor = TemplateProcessing(
        single='[classification] $A [separator]',
        special_tokens=[ # mapping to learned ids
            ('[classification]', tokenizer.token_to_id('[classification]')),
            ('[separator]',      tokenizer.token_to_id('[separator]')),
        ],
    )

    tokenizer.save('tokenizer.json')
    return tokenizer


def load_tokenizer(path='tokenizer.json'):
    ''' loads the trained tokenizer '''
    if not os.path.exists(path):
        return None
    return Tokenizer.from_file(path)


def encode(tokenizer, text):
    ''' encodes a string into a list of tokens '''
    return tokenizer.encode(text).ids


def decode(tokenizer, ids):
    ''' decodes a list of tokens into a string '''
    return tokenizer.decode(ids, skip_special_tokens=True)
