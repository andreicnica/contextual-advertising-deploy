from gensim.models import Word2Vec
import json
import pandas as pd
import treetaggerwrapper as ttw
from utils.functions import extract_tagger_info

EMBEDDING_MODEL_DIR = "dataset/word2vec/french"
#VECTOR_EMBEDDING_MODEL_FILE = EMBEDDING_MODEL_DIR + "/" + "frWiki_no_lem_no_postag_no_phrase_1000_cbow_cut100.bin"
VECTOR_EMBEDDING_MODEL_FILE = EMBEDDING_MODEL_DIR + "/" + "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"

TOP_ADV_KEYTERMS_FILE_RAW = "dataset/top10-keywords-ecommerce.txt"
TOP_ADV_KEYTERMS_FILE_FILTERED = "dataset/top10-keywords-ecommerce-filtered.txt"
EXTRACTED_KEYTERMS_FILE = "dataset/testDataTopPageViews_20Res/testTopPageViewsFiltered_20.json"


EXTRACTOR_ROOT_DIR = "./resources"
TREETAGGER_DIR = EXTRACTOR_ROOT_DIR + "/" + "TreeTagger"

tagger = ttw.TreeTagger(TAGLANG="fr", TAGDIR=TREETAGGER_DIR)


def transform_keyterm_by_vocabulary(keyterm_words, vocabulary):
    term = " ".join(keyterm_words)
    term_len = len(keyterm_words)

    widx = []
    for idx in range(term_len):
        if not keyterm_words[idx] in vocabulary:
            widx.append(idx)

    if not widx:
        return keyterm_words
    else:
        tag_infos = map(extract_tagger_info,
                        tagger.tag_text(term, notagurl=True, notagemail=True, notagip=True, notagdns=True))

        ## Apply a very simple heuristic: if we do not recognize nouns, then drop the whole term.
        ## Otherwise, adjectives, adverbs and numerals can be simply droppped
        non_recognized_word_pos = [tag_infos[idx]['pos'] for idx in widx]
        if "NOM" in non_recognized_word_pos or "NAM" in non_recognized_word_pos:
            return []
        else:
            remaining_words = [keyterm_words[idx] for idx in range(term_len) if not idx in widx]
            return remaining_words


def load_and_filter_adv_keyterms(embedding_model):
    keyterm_list = [[unicode(s, 'utf-8') for s in line.rstrip('\n').split()] for line in open(TOP_ADV_KEYTERMS_FILE_RAW)]
    vocabulary = embedding_model.vocab

    keyterm_dict = {}

    for words in keyterm_list:
        transformed_keyterm = transform_keyterm_by_vocabulary(words, vocabulary)
        if transformed_keyterm:
            k = " ".join(transformed_keyterm)
            keyterm_dict[k] = transformed_keyterm

    return keyterm_dict.values()


def save_filtered_adv_keyterms(keyterm_list, filename):
    with open(filename, "w") as fp:
        keyterm_list = sorted(keyterm_list, key = lambda words : " ".join(words))

        for idx in range (len(keyterm_list)):
            keyterm_words = keyterm_list[idx]
            fp.write(" ".join(keyterm_words).encode('utf-8'))

            if idx < len(keyterm_list) - 1:
                fp.write("\n")


"""
DATASET LOADING FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

def load_adv_keyterms_from_file(filename):
    return [[unicode(s, 'utf-8') for s in line.rstrip('\n').split()] for line in open(filename)]


def load_embedding_model(binary = True):
    return Word2Vec.load_word2vec_format(VECTOR_EMBEDDING_MODEL_FILE, binary = binary)


def load_extracted_keyterms():
    our_dict = {}

    with open(EXTRACTED_KEYTERMS_FILE) as fp:
        test_data = json.load(fp)
        url_keys = test_data.keys()

        for url in url_keys:
            try:
                our_data = json.loads(test_data[url]["keyterms"])
                if our_data["dataIntegrity"]:
                    if len(our_data["keyTerms"]) > 20:
                        our_dict[url] = our_data["keyTerms"][:20]
            except ValueError:
                pass

    return our_dict


"""
MATCHING FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

def match_adv_keyterm(keyterm, embedding_model, min_similarity_threshold):
    pass


def test_top_keyterms(word_list, word2vec_model, filteredKeyterms, k):
    ## TODO - the code below is the old version
    result = map(lambda x: word2vec_model.n_similarity(x, word_list), filteredKeyterms)

    filteredKeytermTuples = [(i, filteredKeyterms[i], result[i]) for i in range(len(filteredKeyterms))]
    sortedKeytermTuples = sorted(filteredKeytermTuples, key=lambda x: x[1], reverse=True)

    return sortedKeytermTuples[:k]