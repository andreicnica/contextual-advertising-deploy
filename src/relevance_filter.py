import pandas as pd

class RelevanceFilter(object):
    def __init__(self, keyterm_feature_df, saved_classifier_file, topk = 10):
        self.keyterm_feature_df = keyterm_feature_df
        self._classifier_file = saved_classifier_file
        self.topk = topk

    def _top_selection(self):
        selection = []

        for row in self.keyterm_feature_df.itertuples(index = False):
            row_set = set(row[0].split())

            subsumed = False
            subsumes = False
            subsumes_index = 0

            for idx in range(len(selection)):
                term_set = set(selection[idx].split())

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
                selection[subsumes_index] = row[0]
            else:
                selection.append(row[0])

            if len(selection) == self.topk:
                break

        return selection



    def select_relevant(self):
        from statsmodels.discrete.discrete_model import LogitResults

        # load classifier model
        model = LogitResults.load("dataset/keyterm-classifier-model-v3.pickle")

        # prepare feature df
        X = self.keyterm_feature_df.copy()
        X = X.drop(['doc_url', "is_url", 'term'], axis = 1)
        X['intercept'] = 1

        self.keyterm_feature_df['relevant_pred'] = model.predict(X)
        self.keyterm_feature_df.sort_values(["relevant_pred", "cvalue"], ascending=[False,False], inplace=True)
        # self.keyterm_feature_df.sort_values(["relevant_pred", "tf"], ascending=[False,False], inplace=True)

        #topk_keyterms = self.keyterm_feature_df[:self.topk]['term'].values
        topk_keyterms = self._top_selection()
        return topk_keyterms

