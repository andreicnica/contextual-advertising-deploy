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


def remove_stopwords(text, lang=LANG_EN):
    """
    :param text: Unicode string or list of words
    :param lang: Language for which to filter stopwords
    :return: Return list of words (in original sequence) from which stopwords are removed or None if the input was not a string or list of strings
    """
    words = []
    if isinstance(text, basestring):
        # split the text into sequence of words
        words = mytokenize(text, lang = lang)
    elif isinstance(text, (list, tuple)):
        words = list(text)

    if words:
        if lang == LANG_EN:
            return [w for w in words if w and w not in stopwords_en]
        elif lang == LANG_FR:
            return [w for w in words if w and w not in stopwords_fr]
        else:
            print "[INFO] Returning empty because of no language."
            return []
    else:
        print "[INFO] Returning empty because of no words for text = ", text
        return []


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


class TextualFeatureExtractor(object):
    def __init__(self, df, urlTokenColum="linkTokens", titleColumn="title", descriptionColumn="resume", textzoneColumn="textZone", anchorColumn="anchors", imgDescColumn="alternateTxt"):
        self.df = df

        if (type(df) != pd.DataFrame):
            raise ValueError("Df is not a pandas.Dataframe")
        else:
            if (urlTokenColum not in df.columns) or (titleColumn not in df.columns) or (descriptionColumn not in df.columns) or (textzoneColumn not in df.columns) or (anchorColumn not in df.columns):
                raise ValueError("Dataframe doesn't contain necessary columns!")

        if (imgDescColumn not in df.columns):
            if ("alternateTxtDesc" in df.columns) and ("alternateTxtZone" in df.columns):
                df["alternateTxt"] = df.alternateTxtZone + df.alternateTxtDesc.apply(lambda x: [x])
            else:
                raise ValueError("Dataframe doesn't contain necessary columns!")

        self.urlTokenColum = urlTokenColum
        self.titleColumn = titleColumn
        self.descriptionColumn = descriptionColumn
        self.textzoneColumn = textzoneColumn
        self.anchorColumn = anchorColumn
        self.imgDescColumn = imgDescColumn

    @staticmethod
    def split_term_grams(term):
        #split only by space (nothing else! like , . ! ? ...)
        grams = []
        sentence = term.split()
        for n in xrange(1, len(sentence)):
            grams = grams + [sentence[i : i+n] for i in xrange(len(sentence)- n+1)]
        return np.ravel(grams)

    #not sensitive to order
    def isURL(self, term, idx):
        if idx not in self.df.index:
            return 0

        line = self.df.loc[idx, self.urlTokenColum]
        if (type(term) is not list) and (type(term) is not np.ndarray):
            ans = line.count(term)
        else:
            ans = sum(map(lambda x: line.count(x), term))
        return ans


    def isTitle(self, term, idx):
        if idx not in self.df.index:
            return 0

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        return self.df.loc[idx, self.titleColumn].count(term)

    def isDescription(self, term, idx):
        if idx not in self.df.index:
            return 0

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        return self.df.loc[idx, self.descriptionColumn].count(term)

    def isAnchor(self, term, idx):
        if idx not in self.df.index:
            return 0

        # if (type(term) is list) or (type(term) is np.ndarray):
        #     term = " ".join(term)
        if isinstance(term, (list, np.ndarray)):
            return sum(map(lambda line: sum(map(lambda x: line.count(x), term)), self.df.loc[idx, self.anchorColumn]))
        elif isinstance(term, basestring):
            return sum(map(lambda line: line.count(term), self.df.loc[idx, self.anchorColumn]))
        else:
            return 0


    def isImgDesc(self, term, idx):
        if idx not in self.df.index:
            return 0

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        return sum(map(lambda x: x.count(term), self.df.loc[idx, self.imgDescColumn]))


    def isFirstParagraph(self, term, idx):
        if idx not in self.df.index:
            return 0

        if len(self.df.loc[idx, self.textzoneColumn]) < 1:
            return 0

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        return sum(map(lambda x: x.count(term), self.df.loc[idx, self.textzoneColumn][0]))

    def isLastParagraph(self, term, idx):
        if idx not in self.df.index:
            return 0

        if len(self.df.loc[idx, self.textzoneColumn]) < 1:
            return 0

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        return sum(map(lambda x: x.count(term), self.df.loc[idx, self.textzoneColumn][-1]))

    def posInDoc(self, term, idx):
        if idx not in self.df.index:
            return None

        if len(self.df.loc[idx, self.textzoneColumn]) < 1:
            return None

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        y = ' '.join(TextualFeatureExtractor.flatten_list(self.df.loc[idx, self.textzoneColumn]))

        if term in y:
            return float(y.index(term))/float(len(y))
        else:
            return None

    def isTextZone(self, term, idx):
        if idx not in self.df.index:
            return 0

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        return sum(map(lambda x: x.count(term), TextualFeatureExtractor.flatten_list(self.df.loc[idx, self.textzoneColumn])))

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
    def lemmas(self):
        return self._term_info['lemma']


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

            self._prop_dict[Term.IS_TITLE] = self.textual_feature_extractor.isTitle(self.lemmas, self.doc.url)
            self._prop_dict[Term.IS_URL] = self.textual_feature_extractor.isURL(self.lemmas, self.doc.url)

            self._prop_dict[Term.IS_DESCRIPTION] = self.textual_feature_extractor.isDescription(self.lemmas, self.doc.url)

            self._prop_dict[Term.IS_FIRST_PARAGRAPH] = self.textual_feature_extractor.isFirstParagraph(self.original, self.doc.url)
            self._prop_dict[Term.IS_LAST_PARAGRAPH] = self.textual_feature_extractor.isLastParagraph(self.original, self.doc.url)

            self._prop_dict[Term.IS_ANCHOR] = self.textual_feature_extractor.isAnchor(self.lemmas, self.doc.url)
            self._prop_dict[Term.IS_IMG_DESC] = self.textual_feature_extractor.isImgDesc(self.lemmas, self.doc.url)

            self._prop_dict[Term.DOC_POSITION] = self.textual_feature_extractor.posInDoc(self.original, self.doc.url)

            if self._prop_dict[Term.DOC_POSITION] is None:
                self._prop_dict[Term. DOC_POSITION] = -1

    ## ================ String and Hashcode Functions ================
    def __str__(self):
        return self._term_info['']

    def __unicode__(self):
        return unicode(self._term_str)

    def __hash__(self):
        return hash(" ".join(self._term_rep))

    def __eq__(self, other):
        if type(other) != Term:
            return False
        else:
            return self.transformed == other.transformed



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