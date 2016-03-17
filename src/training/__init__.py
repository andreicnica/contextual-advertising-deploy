import pandas as pd, treetaggerwrapper as ttw
import json, random
import statsmodels.api as sm

from website_data_extractor import WebsiteDataExtractor
from keyterm_extractor import KeyTermExtractor2
from keyterm_features import KeyTermFeatures2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def read_raw_data(filename):
    df = pd.read_json(filename)
    assert isinstance(df, pd.DataFrame)
    df = df.reset_index()
    df = df.drop('index', axis=1)
    return df

def extract_tagger_info(tag):
    tag_info = tag.split("\t")
    return {'word' : tag_info[0], 'pos' : tag_info[1], 'lemma':tag_info[2]}

def is_meta_keyword(term_lemmas, meta_keyword_lemmas):
    def mysplit(x):
        if " " in x:
            return x.split()
        return x

    split_keyword_lemmas = map(mysplit, meta_keyword_lemmas)

    for tl in term_lemmas:
        if tl != "@card@":
            for kl in split_keyword_lemmas:
                if isinstance(kl, basestring) and tl == kl:
                    return True
                elif isinstance(kl, list) and tl in kl:
                    return True

    return False


def create_raw_dataset(output_filename):
    df_raw = read_raw_data("dataset/preProc2_lower.json")
    page_scraper = WebsiteDataExtractor("dataset/WebsiteElementsPathDef.xml")

    # training dataset is made up only of pages published in 2015
    df_dataset = df_raw.loc[df_raw['dateTime'].map(lambda x: x.year >= 2015)]

    # get all URLs
    all_urls = [x[1] for x in df_dataset['link'].iteritems()]

    # get urls common with grapeshot - these will be our test set
    test_urls = None
    with open("dataset/extracted_terms_grapeshot_common_v3.json") as fp:
        d = json.load(fp, encoding="utf-8")
        test_urls = d.keys()

    print len(test_urls)

    train_urls = [x for x in all_urls if x not in test_urls]
    print len(train_urls)

    train_urls = random.sample(train_urls, 4 * len(test_urls))

    # dataset urls are test + train
    dataset_urls = test_urls + train_urls


    dataset_dict = {}
    idx = 1
    for url in dataset_urls:
        print "[INFO] " + str(idx) + " :: Parsing URL: " + url
        page_data = page_scraper.crawlPage(url)
        dataset_dict[url] = page_data

        idx += 1

    with open(output_filename, mode="w") as fp:
        json.dump(dataset_dict, fp, encoding="utf-8")

    with open("dataset/test_url_list.json", mode="w") as fp:
        json.dump(test_urls, fp, encoding="utf-8")

    with open("dataset/train_url_list.json", mode="w") as fp:
        json.dump(train_urls, fp, encoding="utf-8")

    print "[INFO] Page scraping dataset created."


def create_candidate_term_dataset(dataset_scraped_file, output_filename):
    ## load dataset of scraped page content
    scraped_pages = None
    with open(dataset_scraped_file) as fp:
        scraped_pages = json.load(fp, encoding="utf-8")

    ## load a candidate keyterm extractor
    tagger = ttw.TreeTagger(TAGLANG='fr', TAGDIR=KeyTermExtractor2.TREETAGGER_DIR, )
    candidate_extractor = KeyTermExtractor2(tagger, lang = "french")
    candidate_extractor.initialize()

    dataset_dict = {}
    idx = 1
    for url, data in scraped_pages.items():
        print "[INFO] " + str(idx) + " :: Processing URL " + url
        candidate_extractor.execute(data)
        candidates = candidate_extractor.candidates

        selected_candidates = dict([x for x in candidates.items() if x[1]['cvalue'] > 0])
        dataset_dict[url] = selected_candidates

        candidate_extractor.cleanup()
        idx += 1

    ## save result dataset
    with open(output_filename, mode="w") as fp:
        json.dump(dataset_dict, fp, encoding="utf-8")

    print "[INFO] Candidate term dataset created."


def create_term_features_dataset(scraped_pages_file, candidate_term_file, train_dataset_filename, test_dataset_filename):
    scraped_pages = None
    with open(scraped_pages_file) as fp:
        scraped_pages = json.load(fp, encoding="utf-8")

    candidate_terms = None
    with open(candidate_term_file) as fp:
        candidate_terms = json.load(fp, encoding="utf-8")

    train_urls = None
    with open("dataset/train_url_list.json") as fp:
        train_urls = json.load(fp, encoding="utf-8")

    test_urls = None
    with open("dataset/test_url_list.json") as fp:
        test_urls = json.load(fp, encoding="utf-8")

    ## load a keyterm feature computer
    tagger = ttw.TreeTagger(TAGLANG='fr', TAGDIR=KeyTermExtractor2.TREETAGGER_DIR)
    feature_extractor = KeyTermFeatures2(tagger, lang = "french")

    ## compute selected dataset (i.e. terms with cvalue > 0) and count the size of the would be dataframe
    selected_dataset = []
    selected_size = 0

    for url, candidates in candidate_terms.items():
        page_data_dict = scraped_pages[url]
        #selected_candidates = dict([x for x in candidates.items() if x[1]['cvalue'] > 0])
        selected_candidates = candidates

        selected_size += len(selected_candidates)
        selected_dataset.append((url, page_data_dict, selected_candidates))

    ## create feature_df
    term_df_headers = ['term', 'doc_url', 'cvalue', 'tf', 'is_title', 'is_url',
                        'is_first_par', 'is_last_par', 'is_description',
                        'is_img_desc', 'is_anchor', 'doc_pos', 'relevant']

    feature_df = pd.DataFrame(index = range(selected_size), columns=term_df_headers)

    all_term_features = None
    ct = 1
    for url, page_data_dict, candidates in selected_dataset:
        print "[INFO] " + str(ct) + " :: " + "Computing features for terms of URL " + url
        df = feature_extractor.compute_features(url, page_data_dict, candidates)

        meta_keywords = page_data_dict[WebsiteDataExtractor.KEYWORDS]
        meta_keyword_lemmas = [' '.join([info['lemma'] for info in map(extract_tagger_info, tagger.tag_text(kw, notagurl=True, notagemail=True, notagip=True, notagdns=True))])
                               for kw in meta_keywords]

        relevant_dict = {}
        for idx, term in df['term'].iteritems():
            term_lemma = candidates[term]['lemma_list']
            relevant = is_meta_keyword(term_lemma, meta_keyword_lemmas)
            relevant_dict[term] = relevant

        df['relevant'] = df['term'].map(relevant_dict)

        df_dict = df.to_dict(orient='list')
        if all_term_features is None:
            all_term_features = df_dict
        else:
            for col in all_term_features:
                all_term_features[col].extend(df_dict[col])

        ct += 1

    feature_df = pd.DataFrame(all_term_features)

    ## save train and test feature dataframes
    train_df = feature_df.loc[feature_df['doc_url'].isin(train_urls)]
    test_df = feature_df.loc[feature_df['doc_url'].isin(test_urls)]

    ## save feature data frame
    #feature_df.to_json(train_dataset_filename)
    train_df.to_json(train_dataset_filename)
    test_df.to_json(test_dataset_filename)

    print "[INFO] Keyterm feature datasets created."


class RelevanceClassifier(object):
    TRAIN_DATASET_SIZE = 15000
    TEST_DATASET_SIZE = 4000
    MODEL_STORE_FILE = "dataset/keyterm-classifier-model-updated.pickle"

    def __init__(self, train_df_file, test_df_file):
        self.train_df_file = train_df_file
        self.test_df_file = test_df_file
        self.model = None


    def prepare_training_set(self):
        self.train_df = read_raw_data(self.train_df_file)

        self.pos_train_df = self.train_df[self.train_df['relevant'] == True].sample(self.TRAIN_DATASET_SIZE)
        self.neg_train_df = self.train_df[self.train_df['relevant'] == False].sample(self.TRAIN_DATASET_SIZE)
        self.selected_train_df = pd.concat([self.pos_train_df, self.neg_train_df])

        #self.y, self.X = dmatrices('relevant ~ cvalue + tf + df + tfidf + doc_pos + is_title + is_url + \
        #                           is_anchor + is_description + is_first_par + is_last_par + is_img_desc',
        #                           self.selected_train_df, return_type = "dataframe")
        #self.y = np.ravel(self.y)
        # self.X = self.selected_train_df.drop(['relevant', 'doc_url', 'term'], axis = 1)

        #self.X = self.selected_train_df.drop(['relevant', 'doc_url', 'term', 'df', 'tfidf'], axis = 1)
        print self.selected_train_df.describe()

        self.X = self.selected_train_df.drop(['relevant', 'doc_url', 'term', 'is_url', 'is_first_par', 'is_last_par'], axis = 1)
        self.y = self.selected_train_df['relevant']


    def prepare_test_set(self):
        self.test_df = read_raw_data(self.test_df_file)

        self.pos_test_df = self.test_df[self.test_df['relevant'] == True].sample(self.TEST_DATASET_SIZE)
        self.neg_test_df = self.test_df[self.test_df['relevant'] == False].sample(self.TEST_DATASET_SIZE)
        self.selected_test_df = pd.concat([self.pos_test_df, self.neg_test_df])

        # self.X_test = self.selected_test_df.drop(['relevant', 'doc_url', 'term'], axis = 1)
        #self.X_test = self.selected_test_df.drop(['relevant', 'doc_url', 'term', 'df', 'tfidf'], axis = 1)
        self.X_test = self.selected_test_df.drop(['relevant', 'doc_url', 'term', 'is_url', 'is_first_par', 'is_last_par'], axis = 1)
        self.y_test = self.selected_test_df['relevant']


    def fit(self):
        # self.model = linear_model.LogisticRegression(C=1e3)
        # self.model.fit(self.X, self.y)
        # self.model.score(self.X, self.y)

        X = self.X.copy()
        X['intercept'] = 1

        logit = sm.Logit(self.y, X)
        self.model = logit.fit()
        print self.model.summary()


    def test(self):
        X = self.X_test.copy()
        X['intercept'] = 1

        y_pred = self.model.predict(X)
        print classification_report(self.y_test, (y_pred > 0.5).astype(bool))


def train_relevance_classifier(train_dataset_file, test_dataset_file, model_output_file):
    from statsmodels.discrete.discrete_model import LogitResults
    cl = RelevanceClassifier(train_dataset_file, test_dataset_file)

    print "Preparing training set ..."
    cl.prepare_training_set()

    print "Training model ..."
    cl.fit()

    print "Evaluating model ..."
    cl.prepare_test_set()
    cl.test()

    print "Saving model ..."
    cl.model.save(model_output_file)