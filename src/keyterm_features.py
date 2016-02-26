__author__ = 'alex'

import pandas as pd
import utils.functions as utils
from website_data_extractor import WebsiteDataExtractor

class KeyTermFeatures(object):
    def __init__(self, url, website_data_dict, keyterm_candidate_dict, lang = utils.LANG_EN):
        self.website_data_dict = website_data_dict
        self.keyterm_candidate_dict = keyterm_candidate_dict

        self.url = url
        self.lang = lang


    def compute_features(self):
        ## create document
        doc_text = ''.join(self.website_data_dict[WebsiteDataExtractor.MAIN_TEXT])
        doc = utils.Document(self.url, doc_text, lang=self.lang)

        doc_terms = []
        gram_keys = [('t1gram', 1), ('t2gram', 2), ('t3gram', 3), ('t4gram', 4)]
        for key, length in gram_keys:
            raw_terms = self.keyterm_candidate_dict[key]["term"]
            cvalues = self.keyterm_candidate_dict[key]["cval"]

            for i in range(len(raw_terms)):
                term = utils.Term(raw_terms[i], doc, length, lang=self.lang)
                term.cvalue = cvalues[i]

                # add term to doc
                doc_terms.append(term)

        doc.load_relevant_terms(doc_terms)

        ## compute features of terms in document
        # create df from website_data_dict
        website_data_df = pd.DataFrame.from_dict({self.url : self.website_data_dict}, orient = 'index')
        feature_extractor = utils.TextualFeatureExtractor(website_data_df,
                                        urlTokenColum=WebsiteDataExtractor.URL_TOKENS,
                                        titleColumn=WebsiteDataExtractor.TITLE,
                                        descriptionColumn=WebsiteDataExtractor.SUMMARY,
                                        textzoneColumn=WebsiteDataExtractor.MAIN_TEXT,
                                        anchorColumn=WebsiteDataExtractor.HYPERLINKS,
                                        imgDescColumn=WebsiteDataExtractor.IMAGE_CAPTION)

        # 1) compute TF
        doc.compute_tf()

        term_dataset = []
        for term in doc.relevant_terms:
            # 2) compute linguistic features
            term.set_textual_feature_extractor(feature_extractor)
            term.extract_textual_features()

            term_dataset.append([term.original, doc.url, term.cvalue, term.tf,
                                 term.is_title, term.is_url, term.is_first_par, term.is_last_par, term.is_description,
                                 term.is_img_caption, term.is_anchor, term.doc_position])

        term_df_headers = ['term', 'doc_url', 'cvalue', 'tf', 'is_title', 'is_url',
                        'is_first_par', 'is_last_par', 'is_description',
                        'is_img_desc', 'is_anchor', 'doc_pos']

        return pd.DataFrame(term_dataset, columns=term_df_headers)



class KeyTermFeatures2(object):
    def __init__(self, tagger, lang = utils.LANG_FR):
        self.lang = lang
        self.tagger = tagger

    def compute_features(self, url, website_data_dict, keyterm_candidates):
        ## create document
        doc = utils.Document(url, lang=self.lang)


        doc_terms = []
        for text, info in keyterm_candidates.items():
            d = {'text' : term}.update(info)
            term = utils.Term(d, doc, lang = self.lang)
            doc_terms.append(term)

        doc.load_relevant_terms(doc_terms)

        ## compute features of terms in document
        # create df from website_data_dict and instantiate feature extractor
        website_data_df = pd.DataFrame.from_dict({url : website_data_dict}, orient = 'index')
        feature_extractor = utils.TextualFeatureExtractor(url, website_data_df, self.tagger,
                                        urlTokenColum=WebsiteDataExtractor.URL_TOKENS,
                                        titleColumn=WebsiteDataExtractor.TITLE,
                                        descriptionColumn=WebsiteDataExtractor.SUMMARY,
                                        textzoneColumn=WebsiteDataExtractor.MAIN_TEXT,
                                        anchorColumn=WebsiteDataExtractor.HYPERLINKS,
                                        imgDescColumn=WebsiteDataExtractor.IMAGE_CAPTION)

        term_dataset = []
        for term in doc.relevant_terms:
            # 2) compute linguistic features
            term.set_textual_feature_extractor(feature_extractor)
            term.extract_textual_features()

            term_dataset.append([term.original, doc.url, term.cvalue, term.tf,
                                 term.is_title, term.is_url, term.is_first_par, term.is_last_par, term.is_description,
                                 term.is_img_caption, term.is_anchor, term.doc_position])

        term_df_headers = ['term', 'doc_url', 'cvalue', 'tf', 'is_title', 'is_url',
                        'is_first_par', 'is_last_par', 'is_description',
                        'is_img_desc', 'is_anchor', 'doc_pos']

        return pd.DataFrame(term_dataset, columns=term_df_headers)