from gensim.models import Word2Vec
from sklearn import cluster

import json, time, random
import pandas as pd
import treetaggerwrapper as ttw

from tests.top_adv_keyterm_sim import transform_keyterm_by_vocabulary
from utils.functions import extract_tagger_info
import numpy as np
import matplotlib.pyplot as plt

EMBEDDING_MODEL_DIR = "dataset/word2vec/french"
#VECTOR_EMBEDDING_MODEL_FILE = EMBEDDING_MODEL_DIR + "/" + "frWiki_no_lem_no_postag_no_phrase_1000_cbow_cut100.bin"
VECTOR_EMBEDDING_MODEL_FILE = EMBEDDING_MODEL_DIR + "/" + "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"

GENERATION_NT_CANDIDATES = "./dataset/candidate_keyterms_generation_dataset.json"

EXTRACTOR_ROOT_DIR = "./resources"
TREETAGGER_DIR = EXTRACTOR_ROOT_DIR + "/" + "TreeTagger"

tagger = ttw.TreeTagger(TAGLANG="fr", TAGDIR=TREETAGGER_DIR)


def set_min(min, val):
    if not min:
        return val
    elif min > val:
        return val

    return min

def set_max(max, val):
    if not max:
        return val
    elif max < val:
        return val

    return max


def load_candidate_keyterms(keyterm_candidate_file):
    d = {}
    with open(keyterm_candidate_file) as fp:
        generation_keyterms = json.load(fp)

        for url, keyterms in generation_keyterms.items():
            d[url] = []
            for k in keyterms:
                d[url].append({
                    'term' : k,
                    'lemma_list': keyterms[k]['lemma_list']
                })

    return d


def load_embedding_model(binary = True):
    return Word2Vec.load_word2vec_format(VECTOR_EMBEDDING_MODEL_FILE, binary = binary)


def process_keyterm_clusters(keyterm_candidate_file_trunc):
    keyterm_candidate_file = "./dataset/keyterm_clustering/" + keyterm_candidate_file_trunc + ".json"

    d = load_candidate_keyterms(keyterm_candidate_file)
    embedding_model = load_embedding_model(True)
    vocabulary = embedding_model.vocab

    dataset_list = []
    total_candidates = list(d.items())
    sample_indices = random.sample(xrange(len(total_candidates)), 1000)
    candidate_set = [total_candidates[i] for i in sample_indices]

    for url, keyterms in candidate_set:
        retained = []
        for k in keyterms:
            transformed_lemma = transform_keyterm_by_vocabulary(k['lemma_list'], vocabulary)
            if transformed_lemma:
                retained.append(transformed_lemma)

        X = []
        for kt1 in retained:
            line = []
            for kt2 in retained:
                sim = embedding_model.n_similarity(kt1, kt2)
                line.append(sim)

            X.append(line)

        af = cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
        af.fit(X)
        af_cluster_indices = af.cluster_centers_indices_
        af_labels = af.labels_

        clusters = []

        if not af_cluster_indices is None and len(af_cluster_indices) > 0:
            n_clusters = len(af_cluster_indices)

            min = max = None
            avg = 0

            min_cluster_len = None
            max_cluster_len = None
            avg_cluster_len = 0

            for i in range(len(af_cluster_indices) - 1):
                cluster_center_1 = retained[af_cluster_indices[i]]

                ## compute cluster composition
                cluster_members = []
                for ktIdx in range(len(af_labels)):
                    if af_labels[ktIdx] == i:
                        cluster_members.append(retained[ktIdx])

                cluster_data = {
                    "idx" : i,
                    "center": cluster_center_1,
                    "members": cluster_members,
                    "len": len(cluster_members)
                }

                clusters.append(cluster_data)

                min_cluster_len = set_min(min_cluster_len, cluster_data["len"])
                max_cluster_len = set_max(max_cluster_len, cluster_data["len"])
                avg_cluster_len += cluster_data["len"]

                ## compute cluster distances
                for j in range(i + 1, len(af_cluster_indices)):
                    cluster_center_2 = retained[af_cluster_indices[j]]

                sim = embedding_model.n_similarity(cluster_center_1, cluster_center_2)

                min = set_min(min, sim)
                max = set_max(max, sim)
                avg += sim


            ## compute cluster composition for last cluster
            cluster_center_final = retained[af_cluster_indices[-1]]
            cluster_members = []
            for ktIdx in range(len(af_labels)):
                if af_labels[ktIdx] == len(af_cluster_indices) - 1:
                    cluster_members.append(retained[ktIdx])

            cluster_data = {
                "idx" : len(af_cluster_indices) - 1,
                'center': cluster_center_final,
                "members": cluster_members,
                "len": len(cluster_members)
            }

            clusters.append(cluster_data)

            min_cluster_len = set_min(min_cluster_len, cluster_data["len"])
            max_cluster_len = set_max(max_cluster_len, cluster_data["len"])
            avg_cluster_len += cluster_data["len"]


            ## finalize avereges
            avg /= (n_clusters - 1) * n_clusters / 2
            avg_cluster_len /= len(af_cluster_indices)

            print "Analysed URL: " + url
            dataset_list.append({
                'url': url,
                'nr_clusters': n_clusters,
                'min_dist': min,
                'max_dist': max,
                'avg_dist': avg,
                'min_cluster_len': min_cluster_len,
                'max_cluster_len': max_cluster_len,
                'avg_cluster_len': avg_cluster_len,
                'clusters': clusters
            })

    df = pd.DataFrame.from_records(dataset_list)
    filename = "./dataset/keyterm_clustering/" + keyterm_candidate_file_trunc + "_" + "cluster_analysis" + ".json"
    df.to_json(filename)

    print "DONE!"


    # print "######## Affinity Propagation results for URI: " + url
    # for idx in range(n_clusters):
    #     enumerated_labels = enumerate(af_labels)
    #     class_member_indices = [i for i, j in enumerated_labels if j == idx]
    #
    #     print "Cluster with label: " + str(idx)
    #     print "==========================="
    #     # print class_members
    #     # print class_member_indices
    #     print [retained[i] for i in class_member_indices]
    #     print ""


def get_candidate_keyterms_dataset(output_file, url_list):
    from website_data_extractor import WebsiteDataExtractor
    from keyterm_extractor import KeyTermExtractor2

    data_scraper = WebsiteDataExtractor("dataset/WebsiteElementsPathDef.xml")
    candidate_extractor = KeyTermExtractor2(tagger, lang="french")
    candidate_extractor.initialize()

    dataset_dict = {}

    for link in url_list:
        print "Processing URL: " + link
        data_dict = data_scraper.crawlPage(link)
        candidate_extractor.execute(data_dict)

        dataset_dict[link] = candidate_extractor.candidates
        candidate_extractor.cleanup()

    with open(output_file, "w") as fp:
        json.dump(dataset_dict, fp)


def extract_sample_pausecafe_keyterms():
    with open("dataset/pausecafe_sample_url_list.json") as fp:
        sample_list = json.load(fp)

        get_candidate_keyterms_dataset("dataset/candidate_keyterms_pausecafe_dataset.json", sample_list)


def categorize_pausecafe_df():
    import urlparse

    df_pausecafe = None

    with open("dataset/pausecafe_keyterm_candidate_dataset_cluster_analysis.json") as fp:
        df_pausecafe = pd.read_json(fp)
        df_pausecafe['category'] = df_pausecafe['url'].map(lambda link: urlparse.urlparse(link).path.split("/")[1])

    with open("dataset/pausecafe_keyterm_candidate_dataset_cluster_analysis_categ.json", "w") as fp:
        df_pausecafe.to_json(fp)


def compare_keyterm_clusters():
    df_generation = None
    df_pausecafe = None

    with open("dataset/keyterm_clustering/candidate_keyterms_generation_dataset_cluster_analysis.json") as fp:
        df_generation = pd.read_json(fp)

    with open("dataset/keyterm_clustering/candidate_keyterms_pausecafe_dataset_cluster_analysis.json") as fp:
        df_pausecafe = pd.read_json(fp)

    print "Generation:"
    print df_generation.describe()

    print ""
    print "Pausecafe"
    print df_pausecafe.describe()

    print df_pausecafe.loc[df_pausecafe['max_cluster_len'] == 4,:][['url', 'clusters']].values[0]

    # fig1, ax1 = plt.subplots()
    # df = pd.DataFrame(data=dict(nr_clusters_generation=df_generation.sample(998)["nr_clusters"], nr_clusters_pausecafe=df_pausecafe["nr_clusters"]))
    # # bp_generation = df_generation.boxplot(column=['nr_clusters'], return_type='dict', ax=ax)
    # # bp_pausecafe = df_pausecafe.boxplot(column=['nr_clusters'], return_type='dict', ax=ax)
    # df.boxplot(ax=ax1)
    # fig1.show()

    # fig2, ax2 = plt.subplots()
    #
    # df_pausecafe_low_clusters = df_pausecafe.groupby("category").filter(lambda x : x['nr_clusters'].mean() > 20)
    # df_pausecafe_low_clusters.boxplot(by="category", column="nr_clusters", ax=ax2)
    #
    # for ticklabel in ax2.get_xticklabels():
    #     print ticklabel.set_rotation('vertical')
    #
    # fig2.show()

    # print df.describe()
    # ax = df.boxplot(column=['min_dist', 'max_dist', 'avg_dist'], return_type = 'axes')

def load_cluster_dataset(file_path):
    with open(file_path) as fp:
        top_adv_clusters = np.load(fp)
    return top_adv_clusters

def compute_suggested_adv_cluster_dataset(relative_dataset_filename):
    '''
    This function takes each URI entry from the dataset file, selects only the clusters that have more than 5 elements
    and computes the top 3 most suggested adv keyterm clusters, along with the similarity value
    Similarity is computed between cluster centers
    :param relative_dataset_filename: the name of the cluster analysis dataset file to load, given as relative to the
     dataset/keyterm_clustering folder
    :return:
    '''

    filepath = "dataset/keyterm_clustering/" + relative_dataset_filename + ".json"
    top_adv_clusters_filepath = "dataset/keyterm_clustering/top_adv_keyterm_clusters.dump"

    ## load dataset and embedding model
    print "Loading Embedding model ..."
    embedding_model = load_embedding_model(True)
    vocabulary = embedding_model.vocab

    df = None
    top_adv_clusters = None

    print "Loading datasets ..."
    with open(top_adv_clusters_filepath) as fp:
        top_adv_clusters = np.load(fp)

    with open(filepath) as fp:
        df = pd.read_json(fp)


    ## compute
    result_dataset = []

    print "Starting computation ..."
    for index, row in df.iterrows():
        url = row['url']
        print "Processing clusters for URL: " + url + " ..."

        clusters = row['clusters']
        for cl_data in clusters:
            if cl_data['len'] >= 5:
                similarities = []

                for adv_cl_data in top_adv_clusters:
                    sim = embedding_model.n_similarity(cl_data['center'], adv_cl_data['center'])

                    similarities.append((adv_cl_data['idx'], adv_cl_data['center'], sim))

                similarities.sort(key=lambda x: x[2], reverse=True)
                top3 = similarities[:3]

                result_dataset.append({
                    'url': url,
                    'cl_idx': cl_data['idx'],
                    'cl_center': cl_data['center'],
                    'cl_len': cl_data['len'],
                    'adv1_idx': top3[0][0],
                    'adv1_center': top3[0][1],
                    'adv1_sim': top3[0][2],
                    'adv2_idx': top3[1][0],
                    'adv2_center': top3[1][1],
                    'adv2_sim': top3[1][2],
                    'adv3_idx': top3[2][0],
                    'adv3_center': top3[2][1],
                    'adv3_sim': top3[2][2],
                })

    df_matching = pd.DataFrame.from_records(result_dataset)
    writer = pd.ExcelWriter("dataset/keyterm_clustering/"+ relative_dataset_filename + "_adv_matched" + ".xlsx")
    df_matching.to_excel(writer, "adv_matching")

    writer.save()

    return df_matching

# if __name__ == "__main__":
#     process_keyterm_clusters(GENERATION_NT_CANDIDATES)
