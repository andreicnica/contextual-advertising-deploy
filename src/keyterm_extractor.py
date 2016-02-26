__author__ = 'alex'

import os, shutil, subprocess, codecs, sys
import treetaggerwrapper, csv
import logging, pprint
import nltk.data

from utils.functions import LANG_ABREV
from website_data_extractor import WebsiteDataExtractor


class KeyTermExtractor(object):
    EXTRACTOR_ROOT_DIR = "." + os.sep + "biotex-term-extraction"
    TEMP_PROCESS_DIR = EXTRACTOR_ROOT_DIR + os.sep + "tmp_process"
    TEMP_PROCESS_OUTPUT_DIR = TEMP_PROCESS_DIR + os.sep + "output"
    TEMP_PROCESS_INPUT_FILE = TEMP_PROCESS_DIR + os.sep + "generic-input.txt"

    def __init__(self, website_data_dict):
        self.website_data_dict = website_data_dict
        self.result_dict = {}

    def _initialize(self):
        # 1) first create the temp directory in which the input/output files will be placed
        if os.path.exists(KeyTermExtractor.TEMP_PROCESS_DIR):
            shutil.rmtree(KeyTermExtractor.TEMP_PROCESS_DIR, ignore_errors=True)

        os.makedirs(KeyTermExtractor.TEMP_PROCESS_DIR)
        os.makedirs(KeyTermExtractor.TEMP_PROCESS_OUTPUT_DIR)

        # 2) create generic-input.txt file
        text = ""
        paragraphs = self.website_data_dict.get(WebsiteDataExtractor.MAIN_TEXT)
        if paragraphs is not None and paragraphs:
            text = "\n".join(paragraphs)

        with codecs.open(KeyTermExtractor.TEMP_PROCESS_INPUT_FILE, "w", encoding="utf-8") as fp:
            fp.write(text)

    def _cleanup(self):
        # remove temp process directory
        if os.path.exists(KeyTermExtractor.TEMP_PROCESS_DIR):
            shutil.rmtree(KeyTermExtractor.TEMP_PROCESS_DIR, ignore_errors=True)

    def _collect_result(self):
        import pandas as pd

        dir_path = KeyTermExtractor.TEMP_PROCESS_OUTPUT_DIR

        t1gram_file = dir_path + os.sep + "t1gram.txt"
        t2gram_file = dir_path + os.sep + "t2gram.txt"
        t3gram_file = dir_path + os.sep + "t3gram.txt"
        t4gram_file = dir_path + os.sep + "t4gram.txt"

        if os.path.exists(t1gram_file):
            df_t1gram = pd.read_csv(t1gram_file, sep = ";", header = None, names = ['term', 'check_col', 'cval'], encoding = "utf-8")
            df_t2gram = pd.read_csv(t2gram_file, sep = ";", header = None, names = ['term', 'check_col', 'cval'], encoding = "utf-8")
            df_t3gram = pd.read_csv(t3gram_file, sep = ";", header = None, names = ['term', 'check_col', 'cval'], encoding = "utf-8")
            df_t4gram = pd.read_csv(t4gram_file, sep = ";", header = None, names = ['term', 'check_col', 'cval'], encoding = "utf-8")

            self.result_dict["t1gram"] = df_t1gram[['term', 'cval']].to_dict(orient = 'list')
            self.result_dict["t2gram"] = df_t2gram[['term', 'cval']].to_dict(orient = 'list')
            self.result_dict["t3gram"] = df_t3gram[['term', 'cval']].to_dict(orient = 'list')
            self.result_dict["t4gram"] = df_t4gram[['term', 'cval']].to_dict(orient = 'list')


    def execute(self):
        # 1) initialize
        print "Initializing extractor directory paths ..."
        self._initialize()

        # 2) execute
        print "Performing extraction ..."
        current_dir = os.getcwd()
        os.chdir(KeyTermExtractor.EXTRACTOR_ROOT_DIR)

        try:
            env = dict(os.environ)
            java_command = ['java', '-cp', './classes:./JarBioTexExterne.jar', 'extractor.Extract']
            subprocess.call(java_command, env=env)
        except OSError, e:
            print e
        finally:
            os.chdir(current_dir)

        # 3) collect result
        self._collect_result()

        print "Performing cleanup ..."
        # 4) cleanup
        self._cleanup()



class KeyTermExtractor2(object):
    EXTRACTOR_ROOT_DIR = "." + os.path.sep + "biotex-term-extraction"
    TREETAGGER_DIR = EXTRACTOR_ROOT_DIR + os.path.sep + "TreeTagger"
    POS_PATTERN_DIR = EXTRACTOR_ROOT_DIR + os.path.sep + "patterns"

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

    def __init__(self, lang="english"):
        self.result_dict = {}
        self.lang = lang
        self.tagger = None


    def initialize(self):
        ## setup logging
        root_log = logging.getLogger()
        root_log.setLevel(logging.ERROR)

        stdout_log = logging.StreamHandler(sys.stdout)
        stdout_log.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stdout_log.setFormatter(formatter)
        root_log.addHandler(stdout_log)

        ## initialize the treetagger instance
        lang_abrev = LANG_ABREV.get(self.lang)
        if lang_abrev is None:
            raise ValueError("Unsupported language {}!".format(self.lang))

        self.tagger = treetaggerwrapper.TreeTagger(TAGLANG=lang_abrev, TAGDIR=self.TREETAGGER_DIR)

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

    def _cleanup(self):
        ## close the tagger instance
        if not self.tagger is None:
            self.tagger.__del__()
            self.tagger = None


    def _convert_name_to_index(self, pos_pattern_list):
        base = len(self.pos_tagset) + 1
        return [{'pattern': row['pattern'], 'pattern_nr' : self.convert(row['pattern'], base, self.pos_tagset), 'freq' : row['freq']} for row in pos_pattern_list]

    @staticmethod
    def _extract_tagger_info(tag):
        tag_info = tag.split("\t")
        return {'word' : tag_info[0], 'pos' : tag_info[1], 'lemma':tag_info[2]}


    def execute(self, website_data_dict):
        try:
            ## 1) initialize
            print "Initializing extractor ..."
            self._initialize()

            self.result_dict['t1gram'] = []
            self.result_dict['t2gram'] = []
            self.result_dict['t3gram'] = []
            self.result_dict['t4gram'] = []

            ## 2) execute
            # take each paragraph and split it into sentences
            paragraphs = website_data_dict.get(WebsiteDataExtractor.MAIN_TEXT)
            if not paragraphs is None:
                for p in paragraphs:
                    sentence_list = self.sentence_tokenizer.tokenize(p.strip())
                    for s in sentence_list:
                        tagged_sentence_info = map(self._extract_tagger_info, self.tagger.tag_text(s))
                        clean_sentence_info = [info for info in tagged_sentence_info if info['pos'] in self.pos_tagset]
                        sentence_tag_idx = [self.pos_tagset[info['pos']] for info in clean_sentence_info]

                        selected_term_slices = self.pos_filter.filter(sentence_tag_idx)
                        for term_slice in selected_term_slices:
                            diff = term_slice[1] - term_slice[0]
                            if diff == 1:
                                self.result_dict['t1gram'].append(([info['word'] for info in clean_sentence_info[term_slice[0] : term_slice[1]]],
                                                                   [info['lemma'] for info in clean_sentence_info[term_slice[0] : term_slice[1]]]))
                            elif diff == 2:
                                self.result_dict['t2gram'].append(([info['word'] for info in clean_sentence_info[term_slice[0] : term_slice[1]]],
                                                                   [info['lemma'] for info in clean_sentence_info[term_slice[0] : term_slice[1]]]))
                            elif diff == 3:
                                self.result_dict['t3gram'].append(([info['word'] for info in clean_sentence_info[term_slice[0] : term_slice[1]]],
                                                                   [info['lemma'] for info in clean_sentence_info[term_slice[0] : term_slice[1]]]))
                            else:
                                self.result_dict['t4gram'].append(([info['word'] for info in clean_sentence_info[term_slice[0] : term_slice[1]]],
                                                                   [info['lemma'] for info in clean_sentence_info[term_slice[0] : term_slice[1]]]))


        except Exception as ex:
            logging.getLogger().exception("Failed to perform candidate keyterm extraction.")
        finally:
            self._cleanup()


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