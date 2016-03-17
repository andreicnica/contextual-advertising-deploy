__author__ = 'alex'

import pandas as pd, pprint
import utils.functions as utils
from website_data_extractor import WebsiteDataExtractor


class KeyTermFeatures2(object):
    def __init__(self, tagger, lang = utils.LANG_FR):
        self.lang = lang
        self.tagger = tagger

    def cleanup(self):
        pass

    def compute_features(self, url, website_data_dict, keyterm_candidates):
        ## create document
        doc = utils.Document(url, lang=self.lang)

        doc_terms = []
        for text, info in keyterm_candidates.items():
            d = {'text' : text}
            d.update(info)

            # pprint.pprint(d)
            term = utils.Term(d, doc, lang = self.lang)
            doc_terms.append(term)

        doc.load_relevant_terms(doc_terms)

        ## compute features of terms in document
        # create df from website_data_dict and instantiate feature extractor
        website_data_df = pd.DataFrame.from_dict({url : website_data_dict}, orient = 'index')
        feature_extractor = utils.TextualFeatureExtractor(url, website_data_df, self.tagger,
                                                          urlTokenColumn=WebsiteDataExtractor.URL_TOKENS,
                                                          titleColumn=WebsiteDataExtractor.TITLE,
                                                          descriptionColumn=WebsiteDataExtractor.SUMMARY,
                                                          textzoneColumn=WebsiteDataExtractor.MAIN_TEXT,
                                                          anchorColumn=WebsiteDataExtractor.HYPERLINKS,
                                                          imgDescColumn=WebsiteDataExtractor.IMAGE_CAPTION)

        term_dataset = []
        for term in doc.relevant_terms:
            # compute linguistic features
            term.set_textual_feature_extractor(feature_extractor)
            term.extract_textual_features()

            term_dataset.append([term.original, doc.url, term.cvalue, term.tf,
                                 term.is_title, term.is_url, term.is_first_par, term.is_last_par, term.is_description,
                                 term.is_img_caption, term.is_anchor, term.doc_position])

        term_df_headers = ['term', 'doc_url', 'cvalue', 'tf', 'is_title', 'is_url',
                        'is_first_par', 'is_last_par', 'is_description',
                        'is_img_desc', 'is_anchor', 'doc_pos']

        return pd.DataFrame(term_dataset, columns=term_df_headers)