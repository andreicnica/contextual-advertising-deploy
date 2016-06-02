import string, re, nltk

## GENERAL CONSTANTS
LANG_EN = "english"
LANG_FR = "french"
LANG_ES = "spanish"
LANG_PT = "portuguese"
LANG_IT = "italian"
LANG_DE = "german"

LANG_ABREV = {
    LANG_EN : "en",
    LANG_FR : "fr",
    LANG_ES : "es",
    LANG_PT : "pt",
    LANG_IT : "it",
    LANG_DE : "de"
}



import numpy as np
import pandas as pd

word_split_regex = re.compile(r'[%s\s]+' % re.escape(string.punctuation))
punctuation_regex = re.compile(r"[^\w]+", flags=re.UNICODE)


def mytokenize(text, lang = LANG_EN):
    # decode text to utf-8 if necessary
    if isinstance(text, str):
        text = text.decode("utf-8")

    sentences = nltk.sent_tokenize(text, language = lang)
    return [w for s in sentences for w in punctuation_regex.sub(" ", s).split()]


def read_topkeyterms_from_file(txtfilepath, split=True):
    if not split:
        lines = [line.rstrip('\n') for line in open(txtfilepath)]
    else:
        lines = [line.rstrip('\n').split() for line in open(txtfilepath)]
    return lines

def filter_topkeyterms(keyterm_list, vocabulary):
    keyterm_dict = {}

    for keyterm in keyterm_list:
        aux = [term for term in keyterm if term in vocabulary]

        if aux:
            k = " ".join(aux)

            if not k in keyterm_dict:
                keyterm_dict[k] = aux

    return keyterm_dict.values()


def test_top_keyterms(word_list, word2vec_model, filteredKeyterms, k):
    result = map(lambda x: word2vec_model.n_similarity(x, word_list), filteredKeyterms)

    filteredKeytermTuples = [(i, filteredKeyterms[i], result[i]) for i in range(len(filteredKeyterms))]
    sortedKeytermTuples = sorted(filteredKeytermTuples, key=lambda x: x[1], reverse=True)

    return sortedKeytermTuples[:k]


def remove_stopwords(text, lang=LANG_EN):
    """
    :param text: Unicode string or list of words
    :param lang: Language for which to filter stopwords
    :return: Return list of words (in original sequence) from which stopwords are removed or None if the input was not a string or list of strings
    """
    # words = []
    # if isinstance(text, basestring):
    #     # split the text into sequence of words
    #     words = mytokenize(text, lang = lang)
    # elif isinstance(text, (list, tuple)):
    #     words = list(text)
    #
    # if words:
    #     if lang == LANG_EN:
    #         return [w for w in words if w and w not in stopwords_en]
    #     elif lang == LANG_FR:
    #         return [w for w in words if w and w not in stopwords_fr]
    #     else:
    #         print "[INFO] Returning empty because of no language."
    #         return []
    # else:
    #     print "[INFO] Returning empty because of no words for text = ", text
    #     return []
    pass


def split_into_words(text):
    """
    :param text: Sentence
    :return: list of words split by space
    """
    assert isinstance(text, unicode)
    return text.split()


def clean_string(text):
    import re
    text = u''.join(map(lambda x: x if unicode.isalnum(x) or x == " " else " ", text))
    multiSpacePattern = re.compile(r'\s+')
    text = re.sub(multiSpacePattern, " ", text)
    text.strip()
    return text


def extract_tagger_info(tag):
    #print tag
    tag_info = tag.split("\t")

    ## ensure cardinality lemmas keep the original number
    if tag_info[2] == "@card@":
        return {'word' : tag_info[0], 'pos' : tag_info[1], 'lemma':tag_info[0]}
    else:
        return {'word' : tag_info[0], 'pos' : tag_info[1], 'lemma':tag_info[2]}


class TextualFeatureExtractor(object):
    def __init__(self, url, df, tagger,
                 urlTokenColumn="linkTokens", titleColumn="title", descriptionColumn="resume",
                 textzoneColumn="textZone", anchorColumn="anchors", imgDescColumn="alternateTxt"):
        self.doc_url = url
        self.df = df
        self.tagger = tagger

        self.url_regex = re.compile(ur'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


        if (type(df) != pd.DataFrame):
            raise ValueError("Df is not a pandas.Dataframe")
        else:
            if (urlTokenColumn not in df.columns) or (titleColumn not in df.columns) or (descriptionColumn not in df.columns) or (textzoneColumn not in df.columns) or (anchorColumn not in df.columns):
                raise ValueError("Dataframe doesn't contain necessary columns!")

        if (imgDescColumn not in df.columns):
            if ("alternateTxtDesc" in df.columns) and ("alternateTxtZone" in df.columns):
                df["alternateTxt"] = df.alternateTxtZone + df.alternateTxtDesc.apply(lambda x: [x])
            else:
                raise ValueError("Dataframe doesn't contain necessary columns!")

        self.urlTokenColum = urlTokenColumn
        self.titleColumn = titleColumn
        self.descriptionColumn = descriptionColumn
        self.textzoneColumn = textzoneColumn
        self.anchorColumn = anchorColumn
        self.imgDescColumn = imgDescColumn

        self._setup()


    def _setup(self):
        title_line = self.df.loc[self.doc_url, self.titleColumn]
        anchor_list = self.df.loc[self.doc_url, self.urlTokenColum]
        img_caption_list = self.df.loc[self.doc_url, self.imgDescColumn]
        url_token_list = self.df.loc[self.doc_url, self.urlTokenColum]
        summary_text = self.df.loc[self.doc_url, self.descriptionColumn]

        self.title_lemma_line = ' '.join([info['lemma'] for info in map(extract_tagger_info, self.tagger.tag_text(title_line, notagurl=True, notagemail=True, notagip=True, notagdns=True))])

        self.anchor_lemmas = [' '.join([info['lemma'] for info in map(extract_tagger_info, self.tagger.tag_text(anchor_text, notagurl=True, notagemail=True, notagip=True, notagdns=True))])
                              for anchor_text in anchor_list]

        self.img_lemmas = [' '.join([info['lemma'] for info in map(extract_tagger_info, self.tagger.tag_text(caption, notagurl=True, notagemail=True, notagip=True, notagdns=True))])
                           for caption in img_caption_list]

        self.url_lemmas = [info['lemma'] for token in url_token_list for info in map(extract_tagger_info, self.tagger.tag_text(token, notagurl=True, notagemail=True, notagip=True, notagdns=True))]
        self.summary_lemmas = ' '.join([info['lemma'] for info in map(extract_tagger_info, self.tagger.tag_text(summary_text, notagurl=True, notagemail=True, notagip=True, notagdns=True))])


    @staticmethod
    def split_term_grams(term):
        #split only by space (nothing else! like , . ! ? ...)
        grams = []
        sentence = term.split()
        for n in xrange(1, len(sentence)):
            grams = grams + [sentence[i : i+n] for i in xrange(len(sentence)- n+1)]
        return np.ravel(grams)


    #not sensitive to order
    def isURL(self, term_lemmas):
        if (type(term_lemmas) is not list) and (type(term_lemmas) is not np.ndarray):
            ans = self.url_lemmas.count(term_lemmas)
        else:
            ans = sum(map(lambda x: self.url_lemmas.count(x), term_lemmas))
        return ans


    def isTitle(self, term_lemmas):
        if (type(term_lemmas) is list) or (type(term_lemmas) is np.ndarray):
            term_lemmas = " ".join(term_lemmas)

        return self.title_lemma_line.count(term_lemmas)


    def isDescription(self, term_lemmas):
        if (type(term_lemmas) is list) or (type(term_lemmas) is np.ndarray):
            term_lemmas = " ".join(term_lemmas)

        return self.summary_lemmas.count(term_lemmas)


    def isAnchor(self, term_lemmas):
        # if (type(term) is list) or (type(term) is np.ndarray):
        #     term = " ".join(term)
        if isinstance(term_lemmas, (list, np.ndarray)):
            return sum(map(lambda line: sum(map(lambda x: line.count(x) if self.url_regex.search(x) is None else 0, term_lemmas)), self.anchor_lemmas))

        elif isinstance(term_lemmas, basestring):
            return sum(map(lambda line: line.count(term_lemmas) if self.url_regex.search(term_lemmas) is None else 0, self.anchor_lemmas))

        else:
            return 0


    def isImgDesc(self, term_lemmas):
        if (type(term_lemmas) is list) or (type(term_lemmas) is np.ndarray):
            term_lemmas = " ".join(term_lemmas)

        return sum(map(lambda x: x.count(term_lemmas), self.img_lemmas))


    def isFirstParagraph(self, term_orig):
        if len(self.df.loc[self.doc_url, self.textzoneColumn]) < 1:
            return 0

        if (type(term_orig) is list) or (type(term_orig) is np.ndarray):
            term_orig = " ".join(term_orig)

        return sum(map(lambda x: x.count(term_orig), self.df.loc[self.doc_url, self.textzoneColumn][0]))


    def isLastParagraph(self, term_orig):
        if len(self.df.loc[self.doc_url, self.textzoneColumn]) < 1:
            return 0

        if (type(term_orig) is list) or (type(term_orig) is np.ndarray):
            term_orig = " ".join(term_orig)

        return sum(map(lambda x: x.count(term_orig), self.df.loc[self.doc_url, self.textzoneColumn][-1]))


    def posInDoc(self, term_orig):
        if len(self.df.loc[self.doc_url, self.textzoneColumn]) < 1:
            return None

        if (type(term_orig) is list) or (type(term_orig) is np.ndarray):
            term_orig = " ".join(term_orig)

        y = ' '.join(TextualFeatureExtractor.flatten_list(self.df.loc[self.doc_url, self.textzoneColumn]))
        if term_orig in y:
            return float(y.index(term_orig)) / float(len(y))
        else:
            return -1


    @staticmethod
    def flatten_list(x):
        if isinstance(x, (list, np.ndarray)):
            return [a for i in x for a in TextualFeatureExtractor.flatten_list(i)]
        else:
            return [x]


class Term(object):
    # Information Retrieval Features
    CVAL    = "cvalue"
    TF      = "tf"
    LEN     = "len"

    # Textual Features
    IS_TITLE = "isTitle"
    IS_URL   = "isUrl"
    IS_FIRST_PARAGRAPH  = "firstPar"
    IS_LAST_PARAGRAPH   = "lastPar"
    IS_ANCHOR           = "isAnchor"
    IS_IMG_DESC         = "isImgDesc"
    IS_DESCRIPTION      = "isDescription"
    #IS_TEXTZONE         = "isTextzone"
    DOC_POSITION        = "docPos"
    IS_KEYWORD          = "isKeyword"


    def __init__(self, term_info, doc, lang = LANG_EN):
        self._term_info = term_info
        self.doc = doc
        self.lang = lang

        self._prop_dict = {
            Term.TF     : term_info['tf'],
            Term.CVAL   : term_info['cvalue'],
            Term.LEN    : term_info['len']
        }



    ## ================ Access term representations ================
    @property
    def original(self):
        return self._term_info['text']

    @property
    def words(self):
        return self._term_info['words']

    @property
    def lemma_list(self):
        return self._term_info['lemma_list']

    @property
    def lemma_string(self):
        return self._term_info['lemma_string']

    ## ================ Access Information Retrieval Features ================
    @property
    def tf(self):
        return self._prop_dict.get(Term.TF)

    @tf.setter
    def tf(self, value):
        self._prop_dict[Term.TF] = value

    @property
    def cvalue(self):
        return self._prop_dict.get(Term.CVAL)

    @property
    def term_len(self):
        return self._prop_dict.get(Term.LEN)

    @property
    def is_keyword(self):
        return self._prop_dict.get(Term.IS_KEYWORD)

    @is_keyword.setter
    def is_keyword(self, value):
        self._prop_dict[Term.IS_KEYWORD] = value

    ## ================ Access Textual Features ================
    def set_textual_feature_extractor(self, extractor):
        self.textual_feature_extractor = extractor

    @property
    def is_url(self):
        return self._prop_dict.get(Term.IS_URL)

    @property
    def is_title(self):
        return self._prop_dict.get(Term.IS_TITLE)

    @property
    def is_first_par(self):
        return self._prop_dict.get(Term.IS_FIRST_PARAGRAPH)

    @property
    def is_last_par(self):
        return self._prop_dict.get(Term.IS_LAST_PARAGRAPH)

    @property
    def is_anchor(self):
        return self._prop_dict.get(Term.IS_ANCHOR)

    @property
    def is_img_caption(self):
        return self._prop_dict.get(Term.IS_IMG_DESC)

    @property
    def is_description(self):
        return self._prop_dict.get(Term.IS_DESCRIPTION)

    @property
    def doc_position(self):
        return self._prop_dict.get(Term.DOC_POSITION)


    ## ================ Compute Textual Features ================
    def extract_textual_features(self):
        if self.textual_feature_extractor is not None:

            self._prop_dict[Term.IS_TITLE] = self.textual_feature_extractor.isTitle(self.lemma_string)
            self._prop_dict[Term.IS_URL] = self.textual_feature_extractor.isURL(self.lemma_list)

            self._prop_dict[Term.IS_DESCRIPTION] = self.textual_feature_extractor.isDescription(self.lemma_string)

            self._prop_dict[Term.IS_FIRST_PARAGRAPH] = self.textual_feature_extractor.isFirstParagraph(self.original)
            self._prop_dict[Term.IS_LAST_PARAGRAPH] = self.textual_feature_extractor.isLastParagraph(self.original)

            self._prop_dict[Term.IS_ANCHOR] = self.textual_feature_extractor.isAnchor(self.lemma_string)
            self._prop_dict[Term.IS_IMG_DESC] = self.textual_feature_extractor.isImgDesc(self.lemma_string)

            self._prop_dict[Term.DOC_POSITION] = self.textual_feature_extractor.posInDoc(self.original)


    ## ================ String and Hashcode Functions ================
    def __str__(self):
        return self._term_info['text']

    def __unicode__(self):
        return unicode(self._term_info['text'])

    def __hash__(self):
        return hash(self._term_info['lemma_string'])

    def __eq__(self, other):
        if type(other) != Term:
            return False
        else:
            return self.lemma_string == other.lemma_string



class Document(object):
    def __init__(self, url, lang = LANG_EN):
        self.url = url
        self.lang = lang
        self.relevant_terms = []

    def load_relevant_terms(self, terms):
        self.relevant_terms = terms







"""
:param text:
:return:

util for:

regex match if containing anything else than :: pattern = r'[^a-zA-Z0-9-_]'
regex matches if contains letters a-zA-Z :: pattern = r'[a-zA-Z]'
unicode.strip() -> strip whitespacest
pd.Series.str.split() -> split in words
evaluate list from string -> ast.literal_eval()

#remove from list elem that do not contain letters
df.keywords.apply(lambda ls: [x for x in ls if (re.search('[a-zA-Z]', unidecode(x)))])

#keep unique
df.keywords = df.keywords.apply(np.unique)

df.keywords.apply(lambda ls: [x for x in ls if len(x) > 1])

#match link ending -12341234-0.html
[-[0-9]+]*.html

#----url link----
df.linkTokens = df.linkTokens.str.replace("http://www.generation-nt.com/", "")
df.linkTokens = df.linkTokens.str.replace(r"[-[0-9]+]*.html", "")
df.linkTokens = df.linkTokens.str.split("-")
df.linkTokens.apply(lambda x: x.pop())

#----title-----
df.title = df.title.apply(clean_string)

#----alternateTxtDesc----
df.alternateTxtDesc = df.alternateTxtDesc.apply(clean_string)
df.alternateTxtDesc = df.alternateTxtDesc.apply(unicode.strip)

#----alternateTxtZone----
df.alternateTxtZone = df.alternateTxtZone.apply(lambda x: map(clean_string, x))
df.alternateTxtZone = df.alternateTxtZone.apply(lambda x: map(unicode.strip, x))

#---anchors----
anc = df.anchors
anc = anc.apply(lambda x: map(lambda y: y.values(),x))
anc = anc.apply(lambda x: reduce(lambda o,p: o + p, x) if x else [])
anc = anc.apply(lambda x: filter(lambda y: True if y.startswith("http://www.generation-nt.com/") else False, x))
anc = anc.apply(lambda x: map(lambda y: y.replace("http://www.generation-nt.com/", ""), x))
anc = anc.apply(lambda x: map(lambda y: re.sub(r"[-[0-9]*]*.html", "", y), x))
anc = anc.apply(lambda x: map(lambda y: re.sub(r"[-_+\/]", " ", y), x))
"""