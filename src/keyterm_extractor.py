__author__ = 'alex'

import os, shutil, subprocess, codecs, math
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
            terms[tgram] = {}
            for termLema, termLemaV in tgramVal.iteritems():
                for termL in termLemaV["terms"]:
                    term = " ".join(termL)
                    terms[tgram][term] = {}
                    terms[tgram][term]["termL"] = termL
                    terms[tgram][term]["lemaS"] = termLema
                    terms[tgram][term]["lemaL"] = termLemaV["lemaL"]
                    terms[tgram][term]["tf"] = termLemaV["tf"]
                    terms[tgram][term]["cval"] = termLemaV["cval"]
                    terms[tgram][term]["length"] = len(term)

        return terms


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
        # self._cleanup()
