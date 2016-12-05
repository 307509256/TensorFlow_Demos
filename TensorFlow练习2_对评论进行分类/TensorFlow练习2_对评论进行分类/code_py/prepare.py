import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

org_train_file = 'training.1600000.processed.noemoticon.csv'
org_test_file = 'testdata.manual.2009.06.14.csv'


# extact fields
def use_field(filename_in, filename_out):
    df = pd.read_csv(filename_in,
                     encoding='latin-1',
                     buffer_lines=10000,
                     header=None,
                     names=["opinion", "_1", "_2", "_3", "_4", "text"],
                     usecols=["opinion", "text"],
                     error_bad_lines=False,
                     dtype=np.str)
    df["negative"] = df.opinion.apply(lambda x: 1 if x == "0" else 0)
    df["neutral"] = df.opinion.apply(lambda x: 1 if x == "2" else 0)
    df["positive"] = df.opinion.apply(lambda x: 1 if x == "4" else 0)
    df.reindex(columns=["negative", "neutral", "positive", "text"]) \
        .to_csv(filename_out, encoding="utf-8", sep="|", index=False)


use_field(org_train_file, "train.csv")
use_field(org_test_file, "test.csv")


# make lexicon
def mk_lexicon(filename_in):
    lemmatizer = WordNetLemmatizer()
    df = pd.read_csv(filename_in,
                     encoding="utf-8",
                     sep="|",
                     buffer_lines=10000,
                     header=0,
                     usecols=["text"])
    word_count = {}

    def pipeline_line(x):
        # line=>word=>word_count
        for word in map(lemmatizer.lemmatize, word_tokenize(x.lower())):
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    for i, line in enumerate(df.text):
        if not i % 100000:
            print(i)
        pipeline_line(line)
    lex = [item[0] for item in sorted(word_count.items(), key=lambda x: x[1]) if 100 < item[1] < 100000]
    return lex


lex = mk_lexicon("train.csv")
with open('lexcion.pickle', 'wb') as f:
    pickle.dump(lex, f)
