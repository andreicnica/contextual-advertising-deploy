from gensim.models import Word2Vec
from gensim import matutils
import json
import pandas as pd
import treetaggerwrapper as ttw
from utils.functions import extract_tagger_info
import numpy as np

EMBEDDING_MODEL_DIR = "dataset/word2vec/french"
#VECTOR_EMBEDDING_MODEL_FILE = EMBEDDING_MODEL_DIR + "/" + "frWiki_no_lem_no_postag_no_phrase_1000_cbow_cut100.bin"
VECTOR_EMBEDDING_MODEL_FILE = EMBEDDING_MODEL_DIR + "/" + "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"

TOP_ADV_KEYTERMS_FILE_RAW = "dataset/top10-keywords-ecommerce.txt"
TOP_ADV_KEYTERMS_FILE_FILTERED = "dataset/top10-keywords-ecommerce-filtered.txt"
EXTRACTED_KEYTERMS_FILE = "dataset/testData/testTopPageViewsFiltered_20.json"


EXTRACTOR_ROOT_DIR = "./resources"
TREETAGGER_DIR = EXTRACTOR_ROOT_DIR + "/" + "TreeTagger"

tagger = ttw.TreeTagger(TAGLANG="fr", TAGDIR=TREETAGGER_DIR)


def transform_keyterm_by_vocabulary(keyterm_words, vocabulary):
    term = " ".join(keyterm_words)
    term_len = len(keyterm_words)

    widx = []
    for idx in range(term_len):
        if not keyterm_words[idx] in vocabulary:
            widx.append(idx)

    if not widx:
        return keyterm_words
    else:
        tag_infos = map(extract_tagger_info,
                        tagger.tag_text(term, notagurl=True, notagemail=True, notagip=True, notagdns=True))

        ## Apply a very simple heuristic: if we do not recognize nouns, then drop the whole term.
        ## Otherwise, adjectives, adverbs and numerals can be simply droppped
        non_recognized_word_pos = [tag_infos[idx]['pos'] for idx in widx]
        if "NOM" in non_recognized_word_pos or "NAM" in non_recognized_word_pos:
            return []
        else:
            remaining_words = [keyterm_words[idx] for idx in range(term_len) if not idx in widx]
            return remaining_words


def load_and_filter_adv_keyterms(embedding_model):
    keyterm_list = [[unicode(s, 'utf-8') for s in line.rstrip('\n').split()] for line in open(TOP_ADV_KEYTERMS_FILE_RAW)]
    vocabulary = embedding_model.vocab

    keyterm_dict = {}

    for words in keyterm_list:
        transformed_keyterm = transform_keyterm_by_vocabulary(words, vocabulary)
        if transformed_keyterm:
            k = " ".join(transformed_keyterm)
            keyterm_dict[k] = transformed_keyterm

    return keyterm_dict.values()


def save_filtered_adv_keyterms(keyterm_list, filename):
    with open(filename, "w") as fp:
        keyterm_list = sorted(keyterm_list, key = lambda words : " ".join(words))

        for idx in range (len(keyterm_list)):
            keyterm_words = keyterm_list[idx]
            fp.write(" ".join(keyterm_words).encode('utf-8'))

            if idx < len(keyterm_list) - 1:
                fp.write("\n")


"""
DATASET LOADING FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

def load_adv_keyterms_from_file(filename):
    return [[unicode(s, 'utf-8') for s in line.rstrip('\n').split()] for line in open(filename)]


def load_embedding_model(binary = True):
    return Word2Vec.load_word2vec_format(VECTOR_EMBEDDING_MODEL_FILE, binary = binary)


def load_extracted_keyterms(filepath):
    our_dict = {}

    with open(filepath) as fp:
        test_data = json.load(fp)
        url_keys = test_data.keys()
        for url in url_keys:
            try:
                our_data = json.loads(test_data[url]["keyterms"])
                if our_data["dataIntegrity"]:
                    our_dict[url] = our_data["keyTerms"][:20]
            except ValueError:
                pass

    return our_dict


"""
MATCHING FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

def match_adv_keyterm(keyterm, embedding_model, min_similarity_threshold):
    pass


def test_top_keyterms(word_list, word2vec_model, filteredKeyterms, k):
    ## TODO - the code below is the old version
    result = map(lambda x: word2vec_model.n_similarity(x, word_list), filteredKeyterms)

    filteredKeytermTuples = [(i, filteredKeyterms[i], result[i]) for i in range(len(filteredKeyterms))]
    sortedKeytermTuples = sorted(filteredKeytermTuples, key=lambda x: x[1], reverse=True)

    return sortedKeytermTuples[:k]


def test_clustering():
    from sklearn import cluster

    adv_keyterms = load_adv_keyterms_from_file("dataset/top10-keywords-ecommerce-filtered.txt")
    word2vec_model = load_embedding_model(binary=True)

    X_vec = np.ndarray(shape=(len(adv_keyterms), 200))
    for idx in range(len(adv_keyterms)):
        kt = adv_keyterms[idx]
        v = [word2vec_model[word] for word in kt]
        X_vec[idx] = matutils.unitvec(np.array(v).mean(axis=0))

    X = []
    for kt1 in adv_keyterms:
        line = []
        for kt2 in adv_keyterms:
            sim = word2vec_model.n_similarity(kt1, kt2)
            line.append(sim)

        X.append(line)



    print "Start Affinity Propagation ..."
    af = cluster.AffinityPropagation(affinity="precomputed", damping=0.75)
    af.fit(X)
    print "Finished affinity propagation"

    print "Start KMeans with 75 clusters"
    km = cluster.KMeans(n_clusters=75)
    km.fit(X_vec)
    print "Finish Kmeans"

    af_cluster_indices = af.cluster_centers_indices_
    af_labels = af.labels_
    n_clusters_ = len(af_cluster_indices)

    km_labels = km.labels_

    print "######## Affinity Propagation results ########"
    for idx in range(n_clusters_):
        enumerated_labels = enumerate(af_labels)
        class_members = af_labels == idx
        class_member_indices = [i for i, j in enumerated_labels if j == idx]

        print "Cluster with label: " + str(idx)
        print "==========================="
        # print class_members
        # print class_member_indices
        print [adv_keyterms[i] for i in class_member_indices]
        print ""

    print "######## KMeans results ########"
    for idx in range(n_clusters_):
        enumerated_labels = enumerate(km_labels)
        class_member_indices = [i for i, j in enumerated_labels if j == idx]

        print "Cluster with label: " + str(idx)
        print "==========================="
        # print class_members
        # print class_member_indices
        print [adv_keyterms[i] for i in class_member_indices]
        print ""


def cluster_top_adv_keyterms():
    from sklearn import cluster
    adv_keyterms = load_adv_keyterms_from_file("dataset/top10-keywords-ecommerce-filtered.txt")
    word2vec_model = load_embedding_model(binary=True)

    X = []
    for kt1 in adv_keyterms:
        line = []
        for kt2 in adv_keyterms:
            sim = word2vec_model.n_similarity(kt1, kt2)
            line.append(sim)

        X.append(line)

    print "Start Affinity Propagation ..."
    af = cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
    af.fit(X)
    print "Finished affinity propagation"

    af_cluster_indices = af.cluster_centers_indices_
    af_labels = af.labels_
    n_clusters = len(af_cluster_indices)

    clusters = []

    for i in range(n_clusters):
        cluster_center_1 = adv_keyterms[af_cluster_indices[i]]

        ## compute cluster composition
        cluster_members = []
        for ktIdx in range(len(af_labels)):
            if af_labels[ktIdx] == i:
                cluster_members.append(adv_keyterms[ktIdx])

        cluster_data = {
            "idx" : i,
            "center": cluster_center_1,
            "members": cluster_members,
            "len": len(cluster_members)
        }

        clusters.append(cluster_data)

    with open("dataset/keyterm_clustering/top_adv_keyterm_clusters.dump", "w") as fp:
        np.save(fp, clusters)
        print "Saved top adv keyterm clusters"