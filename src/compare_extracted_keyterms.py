import os, json
import pandas as pd
import treetaggerwrapper as ttw

from collections import OrderedDict
from urlparse import urlparse
from utils.functions import extract_tagger_info

DATASET_FOLDER = "./dataset"
COMPARISON_FOLDER = DATASET_FOLDER + os.path.sep + "testDataTopPageViews_20Res"
COMPARISON_FILE = COMPARISON_FOLDER + os.path.sep + "testTopPageViewsFiltered_20.json"
ERROR_STRING = "Extraction Error"
MAX_TERMS = 20

EXTRACTOR_ROOT_DIR = "." + os.path.sep + "resources"
TREETAGGER_DIR = EXTRACTOR_ROOT_DIR + os.path.sep + "TreeTagger"

tagger = ttw.TreeTagger(TAGLANG="fr", TAGDIR=TREETAGGER_DIR)


def compute_overlap(df_row):
    gs_term_string = df_row['grapeshot_terms']
    our_term_string = df_row['our_terms']

    if gs_term_string == ERROR_STRING or our_term_string == ERROR_STRING:
        return float(0)

    gs_terms = map(lambda x: x.lower(), map(lambda x: x.strip(), gs_term_string.split(",")))
    our_terms = map(lambda x: x.strip(), our_term_string.split(","))

    overlap = 0
    for term in gs_terms:
        for t in our_terms:
            if term in t:
                overlap += 1
                break

    return float(overlap) / len(gs_terms)


def extract_domain(url):
    parsed_uri = urlparse(url)
    domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)

    return domain

def is_noun(term):
    term_split = term.split()
    if len(term_split) > 1:
        return True
    else:
        tag = extract_tagger_info(tagger.tag_text(term, notagurl=True, notagemail=True, notagip=True, notagdns=True)[0])
        if tag["pos"] in ["NOM", "NAM", "ABR"]:
            return True

    return False



def compare_extracted():
    with open(COMPARISON_FILE) as fp:
        test_data = json.load(fp)

        url_keys = test_data.keys()

        gs_dict = {}
        gs_dict_nouns = {}

        our_dict = {}

        for url in url_keys:
            ## save keywords extracted by Grapeshot
            gs_data = json.loads(test_data[url]["GS"], object_pairs_hook=OrderedDict)
            # pprint.pprint(gs_data)

            if "terms" in gs_data:
                aux = gs_data["terms"][0].keys()

                if len(aux) > MAX_TERMS:
                    aux = aux[:MAX_TERMS]

                aux_nouns = [t for t in aux if is_noun(t)]
                gs_dict[url] = ", ".join(aux)
                gs_dict_nouns[url] = ", ".join(aux_nouns)
            else:
                gs_dict[url] = ERROR_STRING
                gs_dict_nouns[url] = ERROR_STRING

            ## save keywords extracted by our approach
            try:
                our_data = json.loads(test_data[url]["keyterms"])
                if our_data["dataIntegrity"]:
                    aux = [info['term'] for info in our_data["keyTerms"]]
                    if len(aux) > MAX_TERMS:
                        aux = aux[:MAX_TERMS]

                    our_dict[url] = ", ".join(aux)
                else:
                    our_dict[url] = ERROR_STRING
            except ValueError:
                our_dict[url] = ERROR_STRING

        ## create dataframes from collected dictionaries
        gs_df = pd.DataFrame.from_dict(gs_dict, orient="index")
        gs_df_nouns = pd.DataFrame.from_dict(gs_dict_nouns, orient="index")

        gs_df.rename(columns={0: "grapeshot_terms"}, inplace=True)
        gs_df_nouns.rename(columns={0: "grapeshot_terms"}, inplace=True)

        our_df = pd.DataFrame.from_dict(our_dict, orient="index")
        our_df.rename(columns={0: "our_terms"}, inplace=True)

        ## create comparison dataframes
        comparison_df = gs_df.merge(our_df, left_index=True, right_index=True)
        comparison_df['overlap'] = comparison_df.apply(compute_overlap, axis=1)

        comparison_df_nouns = gs_df_nouns.merge(our_df, left_index=True, right_index=True)
        comparison_df_nouns['overlap'] = comparison_df_nouns.apply(compute_overlap, axis=1)

        ## add index as column to comparison dataframes to export as Excel tables
        comparison_df = comparison_df.reset_index()
        comparison_df.rename(columns={"index" : "url"}, inplace=True)

        comparison_df_nouns = comparison_df_nouns.reset_index()
        comparison_df_nouns.rename(columns={"index": "url"}, inplace=True)

        comparison_df['site_domain'] = comparison_df.apply(lambda row: extract_domain(row['url']), axis = 1)
        comparison_df_nouns['site_domain'] = comparison_df_nouns.apply(lambda row: extract_domain(row['url']), axis=1)

        return comparison_df, comparison_df_nouns

if __name__ == "__main__":
    comparison_df, comparison_df_nouns = compare_extracted()

    # print comparison_df.describe()

    writer_balanced = pd.ExcelWriter(DATASET_FOLDER + os.path.sep + "comparison_result_balanced.xlsx")
    writer_nouns = pd.ExcelWriter(DATASET_FOLDER + os.path.sep + "comparison_result_nouns.xlsx")

    # writer = pd.ExcelWriter(DATASET_FOLDER + os.path.sep + "comparison_result_unbalanced.xlsx")
    # comparison_df.to_excel(writer, "Comparison")
    # writer.save()

    unique_domains = comparison_df['site_domain'].unique()
    for domain in unique_domains:
        df = comparison_df.loc[comparison_df["site_domain"] == domain, :].copy()
        df.sort_values("overlap", ascending=False, inplace=True)

        df_nouns = comparison_df_nouns.loc[comparison_df_nouns["site_domain"] == domain, :].copy()
        df_nouns.sort_values("overlap", ascending=False, inplace=True)

        parsed_uri = urlparse(domain)
        sheet_name = '{uri.netloc}'.format(uri=parsed_uri)
        df.to_excel(writer_balanced, sheet_name)
        df_nouns.to_excel(writer_nouns, sheet_name)

    writer_balanced.save()
    writer_nouns.save()

