from tqdm import tqdm

from irnlm.utils import save_pickle, write
from irnlm.data.text_tokenizer import tokenize
from irnlm.data.utils import get_possible_morphs, get_possible_pos


morphs = get_possible_morphs()
pos = get_possible_pos()


def extract_syntax(doc): 
    """Extract number of closing nodes for each words of an input sequence. 
    """ 
    ncn = [] 
    morph = []
    pos_ = []
    for sent in doc.sents: 
        parsed_string = sent._.parse_string 
        words = sent.text.split(' ') 
        for token in sent: 
            m = str(morphs.index(str(token.morph))) if str(token.morph)!='' else '0'
            if len(m)==1:
                m = '0' + m
            morph.append(m)
            p = str(pos.index(token.pos_))
            if len(p)==1:
                p = '0' + p
            pos_.append(p)
            word = token.text+')' 
            index = parsed_string.find(word)
            l = len(word) 
            count = 1 
            i = index+l 
            while i<len(parsed_string) and parsed_string[i]==')' :
                count+=1
                i+=1
            ncn.append(str(min(count, 9))) # we take into account a maximum of 9 closing parenthesis
            parsed_string = parsed_string[i:]
    #print(morphs, ncn)
    return [int(''.join(items)) for items in list(zip(ncn, morph, pos_))]

def integral2syntactic(path, nlp, transform_ids, language='english'):
    """Extract syntactic features from the integral text.
    Args:
        - path: list of str (sentences)
        - nlp: Spacy NLP pipelne
        - transform_ids: dict (mapping ids)
        - language: str
    Returns:
        - iterator: list of str (content words)
    """
    iterator = tokenize(path, language=language, with_punctuation=True, convert_numbers=True)
    iterator = [item.strip() for item in iterator]
    #iterator = [' '.join([word.lower() for word in sent.split(' ')]) for sent in iterator]
    docs = [nlp(sent) for sent in iterator if sent !='']

    n = len(docs)
    sentences = [doc.text.split(' ') for doc in tqdm(docs, total=n)]
    activations = [extract_syntax(doc) for doc in tqdm(docs, total=n)]

    save_pickle('./tmp.pkl', activations)
    iterator = []
    for index, activ in tqdm(enumerate(activations), total=n):
        tmp = []
        for i, value in enumerate(activ):
            if value in transform_ids.keys():
                tmp.append(transform_ids[value]+5) # to leave first indexes to special tokens: ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
            else:
                print(value, '-', sentences[index][i])
        iterator.append(tmp)
    iterator = [i for l in iterator for i in l]

    return iterator



# FOR BIG DATASETS
#def text_to_doc(nlp, text, n_split_internal=500, name='_train_', index=0, saving_name='./tmp_doc_syntax.pkl'):
#    """Convert a text 
#    """
#    result = []
#    n = len(text)
#    print('Text length: ', n)
#    for i in tqdm(range(n_split_internal)):
#        data = text[i*n//n_split_internal: (i+1)*n//n_split_internal]
#        try:
#            data = ' . '.join(data.split('.')[:-1])+ ' .'
#        except:
#            print('failed to parse sentences...')
#            data = '.'.join([str(k) for k in data.split('.')][:-1])+ ' .'
#        print('Tokenizing...')
#        data_tmp = tokenize(data, language='english', train=False, with_punctuation=True, convert_numbers=True)
#        print('Parsing...')
#        doc = nlp(' '.join(data_tmp))
#        print('Retrieving syntax...')
#        result.append(extract_syntax(doc))
#    result = [i for l in result for i in l]
#    save_pickle(saving_name, result)
#    return result