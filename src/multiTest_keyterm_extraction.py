import pandas as pd
import urllib2
import os
import sys, argparse, logging
import json
import subprocess
import signal
import httplib

if __name__ == "__main__":

    # if len(sys.argv) < 1:
    #     print "Must specify filepath for the csv with list of urls (csv with header \"links\" for the urls)"
    #     exit()


    parser = argparse.ArgumentParser(prog = 'Keyterm Extraction for list of links', description='Parse input arguments.')
    parser.add_argument('filepath', metavar='filepath', type=str, nargs="?",
               help='test on Grapeshot keyword extractor')

    parser.add_argument('--testOnGS', type=bool, default = False, nargs='?',
                   help='test on Grapeshot keyword extractor')

    parser.add_argument('--keywordExtractorLink', type=str, default = "http://localhost:8000/?link=", nargs='?',
               help='test on Grapeshot keyword extractor')

    parser.add_argument('--GSExtractorLink', type=str, default = "http://ividence.grapeshot.co.uk/standard/channels-json.cgi?url=", nargs='?',
               help='test on Grapeshot keyword extractor')

    parser.add_argument('--resultsPath', type=str, default = "dataset/test_out.json", nargs='?',
               help='test on Grapeshot keyword extractor')



    args = parser.parse_args()
    arg_dict = vars(args)

    filepath = arg_dict['filepath']
    testOnGS = arg_dict['testOnGS']
    keyExtrLink = arg_dict["keywordExtractorLink"]
    GSExtrLink = arg_dict["GSExtractorLink"]
    resultsPath = arg_dict["resultsPath"]

    df = pd.read_csv(filepath)
    results = {}

    f = open(resultsPath, 'w')
    #start server
    f.write("{\n")
    # proc = subprocess.Popen("./src/server_process.py --port 8000 --lang french", shell=True, stdout=file("out.txt", "ab"))
    # os.system("./src/server_process.py --port 8000 --lang french")
    # os.spawnlp(os.P_NOWAIT, "path_to_test.py", "test.py")
    index_nr = 0
    for link in df.links.values:
        print "Extracting Link Nr::" + str(index_nr)

        results = {}

        try:
            out_std = urllib2.urlopen(keyExtrLink + link).read()
        except httplib.BadStatusLine as e:
            out_std = "Reason::error::Link::{}::".format(keyExtrLink + link)


        results["keyterms"] = out_std
        if testOnGS:
            try:
                out_GS = urllib2.urlopen(GSExtrLink + link).read()
            except urllib2.URLError as e:
                out_GS = "Reason::{}::Link::{}::".format(e.reason, GSExtrLink + link)
            results["GS"] = out_GS
        if index_nr == 0:
            f.write("\"{}\": {}".format(link, json.dumps(results)))
        else:
            f.write(",\n\"{}\": {}".format(link, json.dumps(results)))

        index_nr = index_nr + 1

    f.write("}")

    print "Finished. Results in: " + resultsPath

