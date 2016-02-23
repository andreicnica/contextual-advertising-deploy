#from graphlab.cython.context import debug_trace

__author__ = 'andrei'

from lxml import etree
import urllib, sys, re
import xml.etree.ElementTree as ET

try:
    # for beautifulsoup v4
    import bs4
    sys.modules['BeautifulSoup'] = bs4
except ImportError:
    # for beautifulsoup v3
    import BeautifulSoup
    sys.modules['BeautifulSoup'] = BeautifulSoup

from lxml.html.soupparser import fromstring


class WebsiteDataExtractor(object):
    DOMAIN = "domain"
    TITLE = "title"
    SUMMARY = "summary"
    MAIN_TEXT = "mainText"
    HYPERLINKS = "hyperlinks"
    IMAGE_CAPTION = "imageCaption"
    URL_TOKENS = "urlTokens"

    URL_TOKEN_SEPARATOR = "-"

    CONCAT_ANSWERS = ["title"]
    CONCAT_SENTENCES = ["summary"]
    GROUP_BY_CHILDREN = ["mainText"]

    parser = etree.HTMLParser()
    defaultPaths = {}
    customPaths = {}

    def __init__(self, definitionsFile):
        tree = ET.parse(definitionsFile)
        root = tree.getroot()
        for child in root:
            if child.tag == "default":
                for elemPath in child:
                    self.defaultPaths[elemPath.tag] = elemPath.text
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


    def getElementPaths(self, url):
        for domain in self.customPaths.keys():
            if domain in url:
                return self.customPaths[domain]

        return self.defaultPaths

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
            numeric_regex = r"(" + WebsiteDataExtractor.URL_TOKEN_SEPARATOR + "[0-9]+)*"
            token_path = re.sub(numeric_regex, "", token_path)

            tokens = token_path.split(WebsiteDataExtractor.URL_TOKEN_SEPARATOR)
            return tokens


    def crawlPage(self, page):

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

        paths = self.getElementPaths(page)

        pageData = {"urlTokens" : self.tokenizeWebsiteUrl(page)}
        for el, elPattern in paths.iteritems():
            s = tree.xpath(elPattern)

            if el in self.CONCAT_ANSWERS:
                s = self.clean_string_full(" ".join(s))
            elif el in self.CONCAT_SENTENCES:
                s = self.clean_string_partial(" ".join(s))
            elif el in self.GROUP_BY_CHILDREN:
                ls = []
                for aux in s:
                    if (isinstance(aux, etree._Element)):
                        val = aux.xpath('.//text()')
                        for m in val:
                            if (m.isspace() or not m):
                                val.remove(m)
                        if (val):
                            ls = ls + [val]
                s = map(lambda x: " ".join(x), ls)
                s = map(self.clean_string_partial, s)
            else:
                s = map(self.clean_string_partial, s)
                s = filter(None, s)

            pageData[el] = s

        return pageData



#test = WebsiteDataExtractor("dataset/WebsiteElementsPathDef.xml")
#d = test.crawlPage("http://www.generation-nt.com/rechauffement-climatique-ere-glaciaire-retard-actualite-1923734.html")

