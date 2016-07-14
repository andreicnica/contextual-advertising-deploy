#!/usr/bin/env python

"""
HTTP SERVER for running keyterm extractor.
Usage::
    ./server_process [-port <port>] [-lang <lang>]
Send a GET request::
    http://localhost:<port>/?link=<link>
Send a POST request::
    curl -d "link=<link>" http://localhost:<port>
"""

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

import sys, argparse, logging

from pandas.msgpack._unpacker import default_read_extended_type

from website_data_extractor import WebsiteDataExtractor
from keyterm_extractor import KeyTermExtractor2
from keyterm_features import KeyTermFeatures2
from relevance_filter import RelevanceFilter
from keyterm_classification import KeytermClassification

import utils.functions as utils
import urlparse, json, pprint
import treetaggerwrapper as ttw


## Factory for ServerHandlerClass handling http server requests
def makeServerHandlerClass(keytermExtractor):

    class ServerHandlerClass(BaseHTTPRequestHandler, object):
        def __init__(self, *args, **kwargs):
            self.keytermExtractor = keytermExtractor
            super(ServerHandlerClass, self).__init__(*args, **kwargs)

        #global keytermExtractor

        def _set_headers(self):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

        def _set_no_response_headers(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

        def do_GET(self):
            print self.path

            if self.path.startswith("/sien-poc"):
                parsed_qs = urlparse.parse_qs(urlparse.urlparse(self.path).query)
                link = parsed_qs.get('link', None)
                text = parsed_qs.get('text', None)
                linkRecommend = parsed_qs.get("recommend", None)

                terms = self.extractFromLinkOrText(link, text, linkRecommend)
                self._set_headers()
                self.wfile.write(json.dumps(terms))
            else:
                self._set_no_response_headers()
                self.wfile.write("")


        def do_HEAD(self):
            self._set_headers()


        def do_POST(self):
            if self.path.startswith("/sien-poc"):
                content_len = int(self.headers.getheader('content-length', 0))
                post_body = self.rfile.read(content_len)
                link = urlparse.parse_qs(post_body).get('link', None)
                text = urlparse.parse_qs(post_body).get('text', None)
                linkRecommend = urlparse.parse_qs(post_body).get('recommend', None)

                terms = self.extractFromLinkOrText(link, text, linkRecommend)
                self._set_headers()
                self.wfile.write(json.dumps(terms))
            else:
                self._set_no_response_headers()
                self.wfile.write("")


        def extractFromLinkOrText(self, link, text, linkRecommend):
            terms = []
            if not link is None:
                # return JSON response with terms extracted from the link
                terms = self.keytermExtractor.extracTermsFromLink(link[0])

            elif not text is None:
                # return JSON response with terms extracted from text snippet
                terms = self.keytermExtractor.extractTermsFromText(text[0])

            elif not linkRecommend is None:
                # return JSON response with recommandations using base
                terms = self.keytermExtractor.recommendKeytermsForBase(linkRecommend[0])

            return terms


    return ServerHandlerClass


class KeytermServerExtractor(object):
    def __init__(self, port = 8080, lang = utils.LANG_FR, topk = 10):
        print "Initializing Term Extractor Server"

        ## setup server port
        self.port = port
        self.topk = topk

        ## setup keyterm service extraction language
        self.lang = lang
        self.lang_abrev = utils.LANG_ABREV[lang]

        ## setup http request handling classes
        self.server_class = HTTPServer
        self.handler_class = makeServerHandlerClass(self)

        ## setup logging
        ## setup logging
        root_log = logging.getLogger()
        root_log.setLevel(logging.ERROR)

        stdout_log = logging.StreamHandler(sys.stdout)
        stdout_log.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stdout_log.setFormatter(formatter)
        root_log.addHandler(stdout_log)

        ## initialize keyterm extraction service modules
        self._initialize()

    def _initialize(self):
        self.tagger = ttw.TreeTagger(TAGLANG=self.lang_abrev, TAGDIR=KeyTermExtractor2.TREETAGGER_DIR)

        self.data_scraper = WebsiteDataExtractor("dataset/WebsiteElementsPathDef.xml")
        self.candidate_extractor = KeyTermExtractor2(self.tagger, lang = self.lang)
        self.candidate_extractor.initialize()

        self.feature_extractor = KeyTermFeatures2(self.tagger, lang = self.lang)
        #self.relevance_filter = RelevanceFilter("dataset/keyterm-classifier-model-v3.pickle", topk = self.topk)
        #self.relevance_filter = RelevanceFilter("dataset/keyterm-classifier-model-updated.pickle", topk = self.topk)
        self.relevance_filter = RelevanceFilter("dataset/keyterm-classifier-model-general.pickle", topk = self.topk)
        self.keytermClassifier = KeytermClassification(
            classesFile="dataset/top10-keywords-ecommerce-filtered.txt",
            classesClusterPath="dataset/keyterm_clustering/top_adv_keyterm_clusters.dump")

    def _cleanup(self):
        self.tagger = None
        self.data_scraper.cleanup()
        self.candidate_extractor.cleanup()
        self.feature_extractor.cleanup()
        self.relevance_filter.cleanup()


    def runServer(self):
        server_address = ('', self.port)
        httpd = self.server_class(server_address, self.handler_class)

        print 'Starting httpd...'
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            self._cleanup()
            sys.exit(0)
        except Exception as ex:
            logging.getLogger().exception("Error in keyterm extraction!")
            sys.exit(0)


    def extracTermsFromLink(self, link):
        default_return = {
            "available_domains": ["http://www.generation-nt.com/", "http://www.maison.com/",
                                  "http://www.journaldugeek.com/", "http://www.journaldugamer.com/",
                                  "http://www.jdubuzz.com/", "http://news.pixelistes.com/",
                                  "http://www.societe.com/", "http://www.pausecafein.fr/",
                                  "http://worldofwarcraft.judgehype.com/news/",
                                  "http://hearthstone.judgehype.com/news/", "http://diablo3.judgehype.com/news/",
                                  "http://www.judgehype.com/news/",
                                  "http://www.jeuxonline.info", "http://heroes.judgehype.com/news/",
                                  "http://overwatch.judgehype.com/news/",
                                  "http://film-warcraft.judgehype.com/news/", "http://judgehype.com/",
                                  "http://portail.free.fr/", "http://www.planet.fr/",
                                  "http://aliceadsl.closermag.fr/", "http://aliceadsl.lemonde.fr/",
                                  "http://aliceadsl.gqmagazine.fr/"],
            "defaultPath": False, "dataIntegrity":False, "keyTerms":[]}

        try:
            ## 1) Extract webpage data
            print "[INFO] ==== Extracting webpage data ===="
            data_dict = self.data_scraper.crawlPage(link)

            default_return["defaultPath"] = data_dict["defaultPath"]
            default_return["dataIntegrity"] = data_dict["dataIntegrity"]

            if data_dict["defaultPath"] or not data_dict["dataIntegrity"]:
                return default_return

            #pprint.pprint(data_dict)
            ## 2) Extract candidate keyterms
            print "[INFO] ==== Extracting candidate keyterms ===="
            self.candidate_extractor.execute(data_dict)

            # print keyterm_extractor.result_dict
            ## 3) Compute candidate keyterm features
            print "[INFO] ==== Computing candidate keyterm features ===="
            candidate_keyterm_df = self.feature_extractor.compute_features(link, data_dict, self.candidate_extractor.candidates)


            ## 4) Filter for relevancy and output top 10 keyterms
            print "[INFO] ==== Selecting relevant keyterms ===="
            selected_keyterms = self.relevance_filter.select_relevant(candidate_keyterm_df, self.candidate_extractor.candidates)

            # print "[INFO] ==== FINAL SELECTION ====="
            default_return["keyTerms"] = selected_keyterms
            return default_return

        except:
            return default_return



    def extractTermsFromText(self, link):
        default_return = {
            "available_domains": ["http://www.generation-nt.com/", "http://www.maison.com/",
                                  "http://www.journaldugeek.com/", "http://www.journaldugamer.com/",
                                  "http://www.jdubuzz.com/", "http://news.pixelistes.com/",
                                  "http://www.societe.com/", "http://www.pausecafein.fr/",
                                  "http://worldofwarcraft.judgehype.com/news/",
                                  "http://hearthstone.judgehype.com/news/", "http://diablo3.judgehype.com/news/",
                                  "http://www.judgehype.com/news/",
                                  "http://www.jeuxonline.info", "http://heroes.judgehype.com/news/",
                                  "http://overwatch.judgehype.com/news/",
                                  "http://film-warcraft.judgehype.com/news/", "http://judgehype.com/",
                                  "http://portail.free.fr/", "http://www.planet.fr/",
                                  "http://aliceadsl.closermag.fr/", "http://aliceadsl.lemonde.fr/",
                                  "http://aliceadsl.gqmagazine.fr/"],
            "defaultPath": False, "dataIntegrity": False, "keyTerms": []}

        try:
            candidate_keyterms = self.candidate_extractor.execute_with_snippet(text)
            keyterms = self.filter_candidates_from_snippet(candidate_keyterms)

            default_return["keyTerms"] = keyterms
            
            return default_return

        except:
            return default_return

    def recommendKeytermsForBase(self, link):
        default_return = {
            "link_baseline_text": ["title", "description", "keywords", "urlTokens"],
            "keyTerms_recommandations": []}

        try:
            ## 1) Extract webpage data
            print "[INFO] ==== Extracting webpage data USING Specific PathDef===="
            data_dict = self.data_scraper.crawlPage(link, elementsPathDef="baseCluster")

            #Check integrity of list
            if len(data_dict) <= 0:
                return default_return

            # #TEST
            # default_return["crawled_data"] = data_dict

            #Simple extraction of possible terms (not using trained model)
            #Concatanate into text all components with sentence separation
            text_for_analysis = u''
            for key, value in data_dict.iteritems():
                if isinstance(value, basestring):
                    text_for_analysis = text_for_analysis + ". " + value + ". "
                elif isinstance(value, list):
                    text_for_analysis = text_for_analysis + ". ".join(value)

            # #TEST
            # default_return["text_for_analysis"] = text_for_analysis

            #pprint.pprint(data_dict)
            ## 2) Extract candidate keyterms
            print "[INFO] ==== Extracting candidate keyterms ===="
            candidates = self.candidate_extractor.execute_with_snippet(text_for_analysis)

            #TEST
            default_return["keyTerms_candidates"] = candidates

            ## 3) Compute keyterm recommendations comparing cluster centroids
            print "[INFO] ==== Computing keyterm recommendations ===="
            keyterm_recommendations = self.keytermClassifier.match_adv_keyterm_clusters_base(candidates)

            # print "[INFO] ==== FINAL SELECTION ====="
            default_return["keyTerms_recommandations"] = keyterm_recommendations
            return default_return

        except:
            return default_return


    def filter_candidates_from_snippet(self, candidate_keyterms):
        from functools import cmp_to_key

        ordered_keyterms = sorted(candidate_keyterms.itervalues(), key = lambda item: item['cvalue'], reverse = True)
        selected_keyterms = [item for item in ordered_keyterms if item['cvalue'] > 0]

        def pos_cmp(keyterm1, keyterm2):
            if not "NAM" in keyterm1['pos'] and "NAM" in keyterm2['pos']:
                return -1
            elif "NAM" in keyterm1['pos'] and "NAM" not in keyterm2['pos']:
                return 1
            else:
                return 0

        filtered_keyterms = sorted(selected_keyterms, key=cmp_to_key(pos_cmp), reverse=True)

        keyterms = [{'term' : " ".join(t['words']), 'cvalue': t['cvalue'], 'lemma': t['lemma_string'], 'pos_tag': t['pos']} for t in filtered_keyterms]
        return keyterms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = 'Keyterm Extraction Application', description='Parse input arguments for keyterm extraction server application.')
    parser.add_argument('--port', type=int, default = 8080, nargs='?',
                   help='the port on which the server will listen for incoming requests')
    parser.add_argument('--numterms', type=int, default = 10, nargs='?',
                   help='maximum number of terms that will be returned for a webpage')
    parser.add_argument('--lang', type=str, default="french", nargs='?', choices = ['french', 'english'],
                   help='the language used in parsed webpages')

    args = parser.parse_args()
    arg_dict = vars(args)


    port = arg_dict['port']
    topk = arg_dict['numterms']
    lang = arg_dict['lang']

    # create keyterm extractor service
    keytermExtractor = KeytermServerExtractor(port=port, lang=lang, topk=topk)
    print "Test for example with a " + lang + " <link>:: http://localhost:"+ str(port) +"/?link=<link>"

    keytermExtractor.runServer()

