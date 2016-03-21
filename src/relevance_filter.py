from statsmodels.discrete.discrete_model import LogitResults

class RelevanceFilter(object):
    def __init__(self, saved_classifier_file, topk = 10):
        self._classifier_file = saved_classifier_file
        self.topk = topk

        self.model = LogitResults.load(saved_classifier_file)


    def cleanup(self):
        pass


    def _top_selection(self, keyterm_feature_df):
        selection = []

        for row in keyterm_feature_df.itertuples(index = False):
            row_set = set(row[0].split())

            #1) test for identical lemmas
            lemma_exists = False
            for idx in range(len(selection)):
                term_lemma = selection[idx]['lemma']
                if row[3] == term_lemma:
                    lemma_exists = True
                    break

            if lemma_exists:
                continue

            #2) test for subsumption
            subsumed = False
            subsumes = False
            subsumes_index = 0

            for idx in range(len(selection)):
                term_set = set(selection[idx]['term'].split())

                if not (row_set - term_set):
                    subsumed = True
                    break
                elif not (term_set - row_set):
                    subsumes = True
                    subsumes_index = idx
                    break

            if subsumed:
                continue
            elif subsumes:
                selection[subsumes_index] = {'term': row[0], 'cvalue': row[1], 'tf': row[2], 'lemma': row[3], 'pos_tag': row[4] }
            else:
                selection.append({'term': row[0], 'cvalue' : row[1], 'tf': row[2], 'lemma': row[3], 'pos_tag': row[4] })

            if len(selection) == self.topk:
                break

        return selection


    def select_relevant(self, keyterm_feature_df, keyterm_candidates):
        # prepare feature df
        X = keyterm_feature_df.copy()
        #X = X.drop(['doc_url', "is_url", 'term', 'is_first_par', 'is_last_par'], axis = 1)
        X = X.drop(['doc_url', "is_url", 'term', 'is_first_par', 'is_last_par', 'is_title', 'is_description'], axis = 1)
        X['intercept'] = 1

        keyterm_feature_df['relevant_pred'] = self.model.predict(X)
        keyterm_feature_df.sort_values(["relevant_pred", "cvalue"], ascending=[False,False], inplace=True)
        # self.keyterm_feature_df.sort_values(["relevant_pred", "tf"], ascending=[False,False], inplace=True)

        #topk_keyterms = self.keyterm_feature_df[:self.topk]['term'].values
        candidate_lemmas = dict(map(lambda x: (x[0], x[1]['lemma_string']), keyterm_candidates.items()))
        candidate_postags = dict(map(lambda x: (x[0], x[1]['pos']), keyterm_candidates.items()))
        selection_df = keyterm_feature_df.ix[:, ['term', 'cvalue', 'tf']]
        selection_df['lemma'] = selection_df['term'].map(candidate_lemmas)
        selection_df['pos'] = selection_df['term'].map(candidate_postags)

        topk_keyterms = self._top_selection(selection_df)
        topk_keyterms.sort(key = lambda x : x['cvalue'], reverse=True)

        return topk_keyterms

