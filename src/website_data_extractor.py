#from graphlab.cython.context import debug_trace
from itertools import chain

__author__ = 'andrei'

from lxml import etree
import urllib, sys, re, pprint
import xml.etree.ElementTree as ET

class WebsiteDataExtractor(object):
    DOMAIN = "domain"
    TITLE = "title"
    SUMMARY = "summary"
    MAIN_TEXT = "mainText"
    HYPERLINKS = "hyperlinks"
    IMAGE_CAPTION = "imageCaption"
    URL_TOKENS = "urlTokens"
    KEYWORDS = "keywords"

    URL_TOKEN_SEPARATOR = "-"

    CONCAT_ANSWERS = ["title"]
    CONCAT_SENTENCES = ["summary"]
    GROUP_BY_CHILDREN = ["mainText"]
    META_KEYWORDS = ["keywords"]

    parser = etree.HTMLParser()
    defaultPaths = {}
    baseClusterPaths = {}
    customPaths = {}

    def __init__(self, definitionsFile):
        print "INIT Website Data Extractor"

        tree = ET.parse(definitionsFile)
        root = tree.getroot()
        for child in root:
            if child.tag == "default":
                for elemPath in child:
                    self.defaultPaths[elemPath.tag] = elemPath.text
            elif child.tag == "baseCluster":
                for elemPath in child:
                    self.baseClusterPaths[elemPath.tag] = elemPath.text
            else:
                domain = ""
                parsers = {}
                for elemPath in child:
                    if elemPath.tag == "domain":
                        domain = elemPath.text
                    else:
                        parsers[elemPath.tag] = elemPath.text

                if domain:
                    self.customPaths[domain] = parsers

    def cleanup(self):
        pass


    def getElementPaths(self, url):
        for domain in self.customPaths.keys():
            if domain in url:
                return (self.customPaths[domain], False)

        return (self.defaultPaths, True)

    @staticmethod
    def clean_string_partial(text):
        # lowercase only
        text = text.lower()

        # remove multiple spaces (newlines and tab characters)
        multiSpacePattern = re.compile(r'\s+')
        text = re.sub(multiSpacePattern, " ", text)

        # remove inline <script> tags
        return re.sub(r"<script>.*<\/script>", " ", text)

    @staticmethod
    def clean_string_full(text):
        # keep only numbers and letters
        text = u''.join(map(lambda x: x if ( (str.isalnum(x) if isinstance(x, str) else unicode.isalnum(x)) or x == " ") else " ", text))

        # lowercase only
        text = text.lower()

        # remove multiple spaces (newlines and tab characters)
        multiSpacePattern = re.compile(r'\s+')
        text = re.sub(multiSpacePattern, " ", text)

        # remove inline <script> tags
        text = re.sub(r"<script>.*<\/script>", " ", text)

        return text.strip()

    @staticmethod
    def tokenizeWebsiteUrl(url):
        from urlparse import urlparse
        o = urlparse(url)

        if o.path is None or not o.path:
            return []
        else:
            url_path = o.path

            # remove last slash if there is one
            if url_path[-1] == '/':
                url_path = url_path[:-1]

            last_slash_index = url_path.rfind('/')
            url_path = url_path[last_slash_index + 1:]

            # split url by last '.' character, if there is one, and take the first item
            token_path = url_path.split('.')[0]

            # remove numbers from token path
            numeric_regex = ur"(" + WebsiteDataExtractor.URL_TOKEN_SEPARATOR + "[0-9]+)*"
            token_path = re.sub(numeric_regex, "", token_path)

            tokens = token_path.split(WebsiteDataExtractor.URL_TOKEN_SEPARATOR)
            tokens = map(lambda x : x if isinstance(x, unicode) else unicode(x, 'utf-8'), tokens)

            return tokens


    def crawlPage(self, page, elementsPathDef=None):
        try:
            req = urllib.urlopen(page)
        except urllib.HTTPError, e:
            print "Reason::{}::Link::{}::".format(e.reason, page)
            return {}

        if (req.headers.getparam("charset") == "utf-8"):
            tree = etree.fromstring(req.read().decode("utf-8"), self.parser)
        else:
            tree = etree.fromstring(req.read(), self.parser)

        #print "ENCODING::" + req.headers.getparam("charset")

        if elementsPathDef == "baseCluster":
            (paths, defaultPath) = (self.baseClusterPaths, False)
        else:
            (paths, defaultPath) = self.getElementPaths(page)

        pageData = {"urlTokens" : self.tokenizeWebsiteUrl(page), "defaultPath":defaultPath}
        for el, elPattern in paths.iteritems():
            s = tree.xpath(elPattern)

            if len(s) <= 0:
                pageData[el] = []
                continue
            if el in self.CONCAT_ANSWERS:
                conc = u""
                for x in s:
                    if not(isinstance(x, etree._Element)):
                        conc = conc + x + u" "
                s = self.clean_string_partial(conc)
            elif el in self.CONCAT_SENTENCES:
                conc = u""
                for x in s:
                    if not(isinstance(x, etree._Element)):
                        conc = conc + x + u" "
                s = self.clean_string_partial(conc)
            elif el in self.GROUP_BY_CHILDREN:
                ls = []
                for aux in s:
                    if (isinstance(aux, etree._Element)):
                        val = aux.xpath('.//text()[not(parent::script)]')
                        for m in val:
                            if (m.isspace() or not m):
                                val.remove(m)
                        if (val):
                            ls = ls + [val]
                s = map(lambda x: u" ".join(x), ls)
                s = map(self.clean_string_partial, s)
            elif el in self.META_KEYWORDS:
                s = map(lambda x : x if isinstance(x, unicode) else unicode(x, 'utf-8'), s)
                if s:
                    s = s[0].split(",")
            else:
                ## here we treat imageCaptions and urlTokens
                s = map(lambda x : x if isinstance(x, unicode) else unicode(x, 'utf-8'), s)
                s = map(self.clean_string_partial, s)
                s = filter(None, s)

            pageData[el] = s

            pageData["dataIntegrity"] = self.checkIntegrity(pageData)

        return pageData

    @staticmethod
    def checkIntegrity(pageData):
        #Vers_1 percentage of valid data
        # nr_checked = 0.0
        # nr_valid = 0.0
        # for key, value in pageData.iteritems():
        #     if isinstance(value,list) or isinstance(value,str):
        #         nr_checked = nr_checked + 1
        #         if len(value) > 0:
        #             nr_valid = nr_valid + 1
        # return (nr_valid/nr_checked) > 0.4

        must_have = ["title", "mainText"]
        for key in must_have:
            if key not in pageData:
                return False
            else:
                if len(pageData[key]) <= 0:
                    return False

        return True




# test = WebsiteDataExtractor("dataset/WebsiteElementsPathDef.xml")
# d = test.crawlPage("http://www.generation-nt.com/huawei-p9-waterproof-etanche-teasers-photos-actualite-1926204.html")

