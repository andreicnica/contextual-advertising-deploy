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
from website_data_extractor import WebsiteDataExtractor
from keyterm_extractor import KeyTermExtractor2
from keyterm_features import KeyTermFeatures2
from relevance_filter import RelevanceFilter

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

        def do_GET(self):
            link = urlparse.parse_qs(urlparse.urlparse(self.path).query).get('link', None)

            terms = self.extractTermsFromLink(link)

            self._set_headers()

            #send a page containing the list of terms
            #self.wfile.write("<html><body><h1>%s</h1></body></html>" % terms)
            self.wfile.write(json.dumps(terms))

        def do_HEAD(self):
            self._set_headers()

        def do_POST(self):

            content_len = int(self.headers.getheader('content-length', 0))
            post_body = self.rfile.read(content_len)
            link = urlparse.parse_qs(post_body).get('link', None)

            terms = self.extractTermsFromLink(link)

            self._set_headers()
            #send a list of terms
            self.wfile.write(json.dumps(terms))

        def extractTermsFromLink(self, link):
            terms = []
            if link:
                #Run extractor only on first link sent
                terms = self.keytermExtractor.extracTermsFromLink(link[0])
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
        self.relevance_filter = RelevanceFilter("dataset/keyterm-classifier-model-updated.pickle", topk = self.topk)

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
        ## 1) Extract webpage data
        print "[INFO] ==== Extracting webpage data ===="
        data_dict = self.data_scraper.crawlPage(link)
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
        selected_keyterms = self.relevance_filter.select_relevant(candidate_keyterm_df)

        # print "[INFO] ==== FINAL SELECTION ====="
        return selected_keyterms


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
    keytermExtractor.runServer()
