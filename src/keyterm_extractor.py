__author__ = 'alex'


import os, shutil, subprocess, codecs, sys
import csv, re
import logging, math
import nltk.data

from utils.functions import LANG_ABREV, extract_tagger_info, LANG_EN
from website_data_extractor import WebsiteDataExtractor


class KeyTermExtractor2(object):
    EXTRACTOR_ROOT_DIR = "." + os.path.sep + "resources"
    TREETAGGER_DIR = EXTRACTOR_ROOT_DIR + os.path.sep + "TreeTagger"
    POS_PATTERN_DIR = EXTRACTOR_ROOT_DIR + os.path.sep + "patterns"

    url_regex = re.compile(ur'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    email_regex = re.compile(ur"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
    email_prefix_regex = re.compile(ur"(^[a-zA-Z0-9_.+-]+@$)")
    email_suffix_regex = re.compile(ur"(^@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")

    @staticmethod
    def convert(pos_pattern, base, tagset):
            nr = 0
            for tag in pos_pattern:
                nr = nr * base + tagset[tag]

            return nr

    @staticmethod
    def get_number(pos_pattern_idx, base):
        nr = 0
        for tag_idx in pos_pattern_idx:
            nr = nr * base + tag_idx

        return nr

    def __init__(self, tagger, lang = LANG_EN):
        self.candidates = None
        self.lang = lang
        self.tagger = tagger


    def initialize(self):
        ## initialize the treetagger instance
        lang_abrev = LANG_ABREV.get(self.lang)
        if lang_abrev is None:
            raise ValueError("Unsupported language {}!".format(self.lang))

        ## read pos pattern list and POS tagset for selected language
        pos_pattern_filename = self.POS_PATTERN_DIR + os.path.sep + "Patterns_{}_TreeTagger.csv".format(self.lang)
        pos_tagset_filename = self.POS_PATTERN_DIR + os.path.sep + "tagset_{}.txt".format(self.lang)

        self.pos_pattern_list = None
        self.pos_tagset = None

        with open(pos_pattern_filename) as f:
            csv_reader = csv.reader(f, delimiter=';')
            self.pos_pattern_list = [{"pattern" : row[0].strip().split(), "freq" : int(row[1])} for row in csv_reader]

        with open(pos_tagset_filename) as f:
            tags = [t.strip() for t in f.readlines()]
            self.pos_tagset = dict([(tags[idx], (idx + 1)) for idx in range(len(tags))])

        if self.pos_pattern_list is None:
            raise Exception("Could not load POS patterns for selected language {}.".format(self.lang))

        if self.pos_tagset is None:
            raise Exception("Could not load POS tagset for selected language {}.".format(self.lang))

        self.pos_pattern_numbers = self._convert_name_to_index(self.pos_pattern_list)

        ## load sentence tokenizer for selected language
        tokenizer_path = 'tokenizers/punkt/{}.pickle'.format(self.lang)
        self.sentence_tokenizer = nltk.data.load(tokenizer_path)
        if self.sentence_tokenizer is None:
            raise Exception("Could not load sentence tokenizer for selected language {}.".format(self.lang))

        ## initialize POSFilter instance
        self.pos_filter = POSFilter(self.pos_pattern_numbers, self.pos_tagset)


    def cleanup(self):
        self.candidates = None


    def _convert_name_to_index(self, pos_pattern_list):
        base = len(self.pos_tagset) + 1
        return [{'pattern': row['pattern'], 'pattern_nr' : self.convert(row['pattern'], base, self.pos_tagset), 'freq' : row['freq']} for row in pos_pattern_list]


    @staticmethod
    def calcCvalue(gramsDict):

        def isSubList(list, sublist):
            ns = len(sublist)
            for t in range(len(list) - len(sublist) + 1):
                if list[t] == sublist[0]:
                    if list[t+1:t+len(sublist)] == sublist[1:]:
                        return True
            return False


        #!!Parameters will pe modifyed
        nr_grams = len(gramsDict)
        grams = ["t" + str(index) + "grams" for index in range(1, len(gramsDict) + 1)]
        termsLema = {}

        #count freq inside each txgram
        for key, value in gramsDict.iteritems():
            termsLema[key] = {}
            for tple in value:
                termLema = " ".join(tple[1])

                if termLema in termsLema[key]:
                    termsLema[key][termLema]["tf"] += 1
                    if tple[0] not in termsLema[key][termLema]["terms"]:
                        termsLema[key][termLema]["terms"].append(tple[0])
                else:
                    termsLema[key][termLema] = {}
                    termsLema[key][termLema]["terms"] = [tple[0]]
                    termsLema[key][termLema]["lemaL"] = tple[1]
                    termsLema[key][termLema]["pos"] = tple[2]
                    termsLema[key][termLema]["tf"] = 1


        for i in reversed(range(nr_grams)):
            for lema, lemaV in termsLema[grams[i]].iteritems():
                #check bigger terms for candidates
                nr_candidate_terms = 0
                sum_tf_candidate_terms = 0
                for j in range(i + 1, nr_grams):
                    for pt, ptv in termsLema[grams[j]].iteritems():
                        if isSubList(ptv["lemaL"], lemaV["lemaL"]):
                            nr_candidate_terms += 1
                            sum_tf_candidate_terms += ptv["tf"]

                if nr_candidate_terms == 0:
                    lemaV["cval"] = math.log(len(lema), 2) * lemaV["tf"]
                else:
                    lemaV["cval"] = math.log(len(lema), 2) * (lemaV["tf"] - sum_tf_candidate_terms / nr_candidate_terms)

        terms = {}
        for tgram, tgramVal in termsLema.iteritems():
            # terms[tgram] = {}
            for termLema, termLemaV in tgramVal.iteritems():
                for termL in termLemaV["terms"]:
                    term = " ".join(termL)
                    terms[term] = {
                        "words" : termL,
                        "lemma_string" : termLema,
                        "lemma_list" : termLemaV["lemaL"],
                        "pos": termLemaV["pos"],
                        "tf" : termLemaV["tf"],
                        "cvalue" : termLemaV["cval"],
                        "len" : len(term)
                    }

        return terms


    def execute(self, website_data_dict):
        grams_dict = {
            't1grams' : [],
            't2grams' : [],
            't3grams' : [],
            't4grams' : []
        }

        ## 2) execute
        # take each paragraph and split it into sentences. Paragraphs include the TITLE and the SUMMARY sentences as well
        title = website_data_dict.get(WebsiteDataExtractor.TITLE)
        summary = website_data_dict.get(WebsiteDataExtractor.SUMMARY)
        mainText = website_data_dict.get(WebsiteDataExtractor.MAIN_TEXT)

        paragraphs = []
        if not title is None and title:
            paragraphs.append(title)

        if not summary is None and summary:
            paragraphs.append(summary)

        if not mainText is None and mainText:
            paragraphs.extend(mainText)

        #paragraphs = website_data_dict.get(WebsiteDataExtractor.MAIN_TEXT)
        #if not paragraphs is None:

        for p in paragraphs:
            sentence_list = self.sentence_tokenizer.tokenize(p.strip())
            for s in sentence_list:
                # check that the sentence is not empty
                if s:
                    tagged_sentence_info = map(extract_tagger_info, self.tagger.tag_text(s, notagurl=True, notagemail=True, notagip=True, notagdns=True))
                    clean_sentence_info = [info for info in tagged_sentence_info if info['pos'] in self.pos_tagset]
                    sentence_tag_idx = [self.pos_tagset[info['pos']] for info in clean_sentence_info]

                    selected_term_slices = self.pos_filter.filter(sentence_tag_idx)
                    for term_slice in selected_term_slices:
                        diff = term_slice[1] - term_slice[0]
                        gram_tuple = ([info['word'] for info in clean_sentence_info[term_slice[0] : term_slice[1]]],
                                     [info['lemma'] for info in clean_sentence_info[term_slice[0] : term_slice[1]]],
                                     [info['pos'] for info in clean_sentence_info[term_slice[0] : term_slice[1]]])

                        term_text = "".join(gram_tuple[0])
                        if self.url_regex.search(term_text) is None and self.email_regex.search(term_text) is None \
                            and self.email_prefix_regex.search(term_text) is None and self.email_suffix_regex.search(term_text) is None \
                            and not u"\u00A9" in term_text:
                            if diff == 1:
                                grams_dict['t1grams'].append(gram_tuple)
                            elif diff == 2:
                                grams_dict['t2grams'].append(gram_tuple)
                            elif diff == 3:
                                grams_dict['t3grams'].append(gram_tuple)
                            else:
                                grams_dict['t4grams'].append(gram_tuple)

        self.candidates = self.calcCvalue(grams_dict)



    def execute_with_snippet(self, text):
        grams_dict = {
            't1grams': [],
            't2grams': [],
            't3grams': [],
            't4grams': []
        }

        if not isinstance(text, unicode):
            text = unicode(text, "utf-8")

        sentence_list = self.sentence_tokenizer.tokenize(text.strip())

        for s in sentence_list:
            # check that the sentence is not empty
            if s:
                tagged_sentence_info = map(extract_tagger_info,
                                           self.tagger.tag_text(s, notagurl=True, notagemail=True, notagip=True,
                                                                notagdns=True))
                clean_sentence_info = [info for info in tagged_sentence_info if info['pos'] in self.pos_tagset]
                sentence_tag_idx = [self.pos_tagset[info['pos']] for info in clean_sentence_info]

                selected_term_slices = self.pos_filter.filter(sentence_tag_idx)
                for term_slice in selected_term_slices:
                    diff = term_slice[1] - term_slice[0]
                    gram_tuple = ([info['word'] for info in clean_sentence_info[term_slice[0]: term_slice[1]]],
                                  [info['lemma'] for info in clean_sentence_info[term_slice[0]: term_slice[1]]],
                                  [info['pos'] for info in clean_sentence_info[term_slice[0]: term_slice[1]]])

                    term_text = "".join(gram_tuple[0])
                    if self.url_regex.search(term_text) is None and self.email_regex.search(term_text) is None \
                            and self.email_prefix_regex.search(term_text) is None and self.email_suffix_regex.search(
                        term_text) is None \
                            and not u"\u00A9" in term_text:
                        if diff == 1:
                            grams_dict['t1grams'].append(gram_tuple)
                        elif diff == 2:
                            grams_dict['t2grams'].append(gram_tuple)
                        elif diff == 3:
                            grams_dict['t3grams'].append(gram_tuple)
                        else:
                            grams_dict['t4grams'].append(gram_tuple)


        candidates = self.calcCvalue(grams_dict)
        return candidates



class POSFilter(object):
    """
    The class performs filtering of candidate keyterms based on patterns of POS tags.
    """
    def __init__(self, pos_patterns, tagset):
        '''
        :param pos_patterns:  list where each element is itself a list representing the sequence of POS tags making up a pattern
        :param tagset: complete set of tags for the chosen language on which this pattern filter will run
        '''
        self.tagset = tagset
        self.pos_patterns = self._order_patterns(pos_patterns)
        self.max_pattern_len = len(self.pos_patterns[-1])

        ## call initialization
        self._initialize()


    def _order_patterns(self, pos_patterns):
        return sorted(pos_patterns, key = lambda row : len(row['pattern']))


    def _initialize(self):
        plen = 1
        d = {}
        for p in self.pos_patterns:
            if len(p['pattern']) == plen:
                if not plen in d:
                    d[plen] = set([p['pattern_nr']])
                else:
                    d[plen].add(p['pattern_nr'])
            else:
                plen = len(p['pattern'])
                if not plen in d:
                    d[plen] = set([p['pattern_nr']])
                else:
                    d[plen].add(p['pattern_nr'])

        self.clustered_patterns = d.items()


    def filter(self, sentence_pos_idx):
        selected = []

        for patterns_by_len in self.clustered_patterns:
            plen = patterns_by_len[0]
            #print "Running for patterns of length: " + str(plen)

            pattern_set = patterns_by_len[1]
            #pprint.pprint(pattern_set)

            ## generate all ngrams of sentence_pos_list of length plen
            grams = [(i, i + plen, KeyTermExtractor2.get_number(sentence_pos_idx[i : i + plen], len(self.tagset) + 1))
                             for i in xrange(len(sentence_pos_idx) - plen + 1)]

            for gram in grams:
                if gram[2] in pattern_set:
                    selected.append(gram)

        return selected
