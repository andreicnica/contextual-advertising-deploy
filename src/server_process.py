#!/usr/bin/env python

"""
HTTP SERVER for running keyterm extractor.
Usage::
    ./server_process [<port>]
Send a GET request::
    http://localhost:<port>/?link=<link>
Send a POST request::
    curl -d "link=<link>" http://localhost:<port>
"""

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer

import sys
from website_data_extractor import WebsiteDataExtractor
from keyterm_extractor import KeyTermExtractor
from keyterm_features import KeyTermFeatures
from relevance_filter import RelevanceFilter
import utils.functions as utils
import urlparse, json, pprint


#Handler for http server reques
class ServerHandlerClass(BaseHTTPRequestHandler):
    global keytermExtractor

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
            terms = keytermExtractor.extracTermsFromLink(link[0])
        return terms

class KeytermServerExtractor(object):
    data_extractor = WebsiteDataExtractor("dataset/WebsiteElementsPathDef.xml")

    def __init__(self):
        print "INIT TERM EXtractor"

    def runServer(self, server_class=HTTPServer, handler_class=ServerHandlerClass, port=8080):
        server_address = ('', port)
        httpd = server_class(server_address, handler_class)

        print 'Starting httpd...'
        httpd.serve_forever()

    def extracTermsFromLink(self, link):
        ## 1) Extract webpage data
        # print "[INFO] ==== Extracting webpage data ===="
        data_dict = self.data_extractor.crawlPage(link)
        pprint.pprint(data_dict)

        ## 2) Extract candidate keyterms
        # print "[INFO] ==== Extracting candidate keyterms ===="
        keyterm_extractor = KeyTermExtractor(data_dict)
        keyterm_extractor.execute()

        # print keyterm_extractor.result_dict
        ## 3) Compute candidate keyterm features
        # print "[INFO] ==== Computing candidate keyterm features ===="
        keyterm_feat = KeyTermFeatures(link, data_dict, keyterm_extractor.result_dict, lang=utils.LANG_FR)
        candidate_keyterm_df = keyterm_feat.compute_features()


        ## 4) Filter for relevancy and output top 10 keyterms
        # print "[INFO] ==== Selecting relevant keyterms ===="
        relevance_filter = RelevanceFilter(candidate_keyterm_df, "dataset/keyterm-classifier-model-v2.pickle", topk=10)
        selected_keyterms = relevance_filter.select_relevant()

        # print "[INFO] ==== FINAL SELECTION ====="
        return selected_keyterms

#initial data
keytermExtractor = KeytermServerExtractor()


if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        keytermExtractor.runServer(port=int(argv[1]))
    else:
        keytermExtractor.runServer()
