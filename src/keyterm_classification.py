from statsmodels.discrete.discrete_model import LogitResults
import gensim
from gensim.models import Word2Vec
import json
import pandas as pd
import treetaggerwrapper as ttw
from utils.functions import extract_tagger_info
from difflib import SequenceMatcher
import numpy as np
import collections

# EMBEDDING_MODEL_DIR = "dataset/word2vec/french"
#VECTOR_EMBEDDING_MODEL_FILE = EMBEDDING_MODEL_DIR + "/" + "frWiki_no_lem_no_postag_no_phrase_1000_cbow_cut100.bin"
# VECTOR_EMBEDDING_MODEL_FILE = EMBEDDING_MODEL_DIR + "/" + "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"

# TOP_ADV_KEYTERMS_FILE_RAW = "dataset/top10-keywords-ecommerce.txt"
# TOP_ADV_KEYTERMS_FILE_FILTERED = "dataset/top10-keywords-ecommerce-filtered.txt"
# EXTRACTED_KEYTERMS_FILE = "dataset/testDataTopPageViews_20Res/testTopPageViewsFiltered_20.json"

EXTRACTOR_ROOT_DIR = "./resources"
TREETAGGER_DIR = EXTRACTOR_ROOT_DIR + "/" + "TreeTagger"

class KeytermClassification(object):
    classes = {}
    tagger = ttw.TreeTagger(TAGLANG="fr", TAGDIR=TREETAGGER_DIR)
    classes_filtered = {}
    classes_filtered_keywords = []

    def __init__(self, classes=None, classesFile=None,
                 modelPath="dataset/frWiki_no_lem_no_postag_no_phrase_1000_cbow_cut200.bin", modelBinary=True):

        if not(classes) and not(classesFile):
            print ""

        if not(classes):
            classes = self.load_adv_keyterms_from_file(classesFile)

        self.model = gensim.models.Word2Vec.load_word2vec_format(modelPath, binary=modelBinary)
        self._preProcessClasses(classes)

    def _searchInModel(self, keyword):
        return keyword in self.model.vocab

    def _checkSimilarty(self, cl, keyterm):
        return self.model.n_similarity(cl, keyterm)

    def _preProcessClasses(self, classes):
        for cl in classes:
            class_name = " ".join(cl)
            self.classes[class_name] = {}
            self.classes[class_name]["keyterm"] = cl
            self.classes[class_name]['filtered'] = self.transform_keyterm_by_vocabulary(cl)
            filtered = " ".join(self.classes[class_name]['filtered'])
            if filtered:
                if filtered in self.classes_filtered:
                    self.classes_filtered[filtered]["keyterms"].append(cl)
                else:
                    self.classes_filtered[filtered] = {}
                    self.classes_filtered[filtered]["keyterms"] = [cl]
                    self.classes_filtered[filtered]["class"] = self.classes[class_name]['filtered']
                    self.classes_filtered_keywords.append(self.classes[class_name]['filtered'])

    @staticmethod
    def load_adv_keyterms_from_file(filename):
        return [[unicode(s, 'utf-8') for s in line.rstrip('\n').split()] for line in open(filename)]

    def transform_keyterm_by_vocabulary(self, keyterm_words):
        term = " ".join(keyterm_words)
        term_len = len(keyterm_words)

        widx = []
        for idx in range(term_len):
            if not self._searchInModel(keyterm_words[idx]):
                widx.append(idx)

        if not widx:
            return keyterm_words
        else:
            tag_infos = map(extract_tagger_info,
                            self.tagger.tag_text(term, notagurl=True, notagemail=True, notagip=True, notagdns=True))

            ## Apply a very simple heuristic: if we do not recognize nouns, then drop the whole term.
            ## Otherwise, adjectives, adverbs and numerals can be simply droppped
            non_recognized_word_pos = [tag_infos[idx]['pos'] for idx in widx]
            if "NOM" in non_recognized_word_pos or "NAM" in non_recognized_word_pos:
                return []
            else:
                remaining_words = [keyterm_words[idx] for idx in range(term_len) if not idx in widx]
                return remaining_words

    def match_adv_keyterm(self, keyterm,
                          min_similarity_threshold=0, min_diff_distance=0.90,
                          top=5):

        if not isinstance(keyterm, basestring):
            raise ValueError("!!")

        keyterm_key = unicode(keyterm)
        keyterm = keyterm_key.split()

        result = []

        #Check if keyterm Ratcliff-Obershelp distance to one of the classes is above distance
        #If any exists return the classes
        diff_distance = np.array(map(lambda x: SequenceMatcher(None, x, keyterm_key).ratio(), self.classes.keys()))
        if max(diff_distance) >= min_diff_distance:
            selected = np.where(diff_distance >= min_diff_distance)
            result = zip(np.array(self.classes.keys())[selected], np.array(diff_distance)[selected])
            result.sort(key=lambda tup: tup[1])
            result = (np.array(result))[::-1]
            return result[:top]

        #Further check distance by similarity

        keyterm_filtered = self.transform_keyterm_by_vocabulary(keyterm)

        if not keyterm_filtered:
            return []

        similarity = np.array(map(lambda cl: self._checkSimilarty(cl, keyterm_filtered), self.classes_filtered_keywords))

        selected = np.where(similarity > min_similarity_threshold)
        result = zip(np.array(
            map(lambda x: " ".join(x), self.classes_filtered_keywords))[selected], np.array(similarity)[selected])

        result.sort(key=lambda tup: tup[1])

        result = (np.array(result))[::-1]
        return result[:top]
