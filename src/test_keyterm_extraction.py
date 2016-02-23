__author__ = "alex"

import sys
from website_data_extractor import WebsiteDataExtractor
from keyterm_extractor import KeyTermExtractor
# from keyterm_features import KeyTermFeatures
# from keyterm_classifier import RelevanceFilter
# import utils.functions as utils


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: python test_keyterm_extraction.py <webpage_url>"
    else:
        url = sys.argv[1]

        ## 1) Extract webpage data
        print "[INFO] ==== Extracting webpage data ===="
        data_extractor = WebsiteDataExtractor("dataset/WebsiteElementsPathDef.xml")
        data_dict = data_extractor.crawlPage(url)

        # ## 2) Extract candidate keyterms
        # print "[INFO] ==== Extracting candidate keyterms ===="
        # keyterm_extractor = KeyTermExtractor(data_dict)
        # keyterm_extractor.execute()
        #
        # # print keyterm_extractor.result_dict
        # ## 3) Compute candidate keyterm features
        # print "[INFO] ==== Computing candidate keyterm features ===="
        # keyterm_feat = KeyTermFeatures(url, data_dict, keyterm_extractor.result_dict, lang=utils.LANG_FR)
        # candidate_keyterm_df = keyterm_feat.compute_features()
        #
        #
        # ## 4) Filter for relevancy and output top 10 keyterms
        # print "[INFO] ==== Selecting relevant keyterms ===="
        # relevance_filter = RelevanceFilter(candidate_keyterm_df, "dataset/keyterm-classifier-model-v2.pickle", topk=10)
        # selected_keyterms = relevance_filter.select_relevant()
        #
        # print "[INFO] ==== FINAL SELECTION ====="
        # print "\n".join(selected_keyterms)
