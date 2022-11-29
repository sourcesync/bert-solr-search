import bz2
import time
import requests
import json
import traceback
import pandas as pd
import sys
import os

# Determine if we also send tokens
HYBRID = True

# Set to True to print a bunch of debugging stuff
DEBUG = False

# The compressed text corpus
input_file = '../long_abstracts_en.ttl.bz2'

# Number of vectors to pack into a request payload
PACK    = 25

# Change this constant to vary the number of indexed abstracts
# set to -1 to index all
MAX_DOCS = 1000000

from data_utils import parse_dbpedia_data, SearchEngine, EmbeddingModel, enrich_doc_with_vectors
from transformers import AutoTokenizer

# Figure out if we are skipping past uploaded payloads
SKIP_TO = 0
if os.path.exists("results.csv.last"):
    df = pd.read_csv( "results.csv.last" )
    SKIP_TO = df['idx'].max()
    print("SKIP_TO=", SKIP_TO)
    #sys.exit(0)

# Read the input compressed file as is, without decompressing.
# Though disks are cheap, isn't it great to save them?
print("Reading BZ2 file")
source_file = bz2.BZ2File(input_file, "r")

# Get a document iterator.
print("Computing vectors from scratch")
docs_iter = parse_dbpedia_data(source_file, MAX_DOCS)
docs_iter = enrich_doc_with_vectors(docs_iter, EmbeddingModel.HUGGING_FACE_SENTENCE, SearchEngine.ELASTICSEARCH)

# Create a tokenizer
tokenizer = AutoTokenizer.from_pretrained('transfo-xl-wt103')

results = []

def upsert(idx, vecs, toks):
    headers = { \
        'Content-Type':'application/json', \
        #'Api-Key':'5cfe85a2-4444-405a-b5f3-a1fb8207be23' \
        'Api-Key':'658be878-381f-4835-9723-cf9b6bb12ce1'
    }
    vectors = []
    for i, vec in enumerate(vecs):
        idint = idx+i
        vectors.append( { \
            'id': str(idint), \
            'metadata': {} if not HYBRID else {'tokens':toks[i]}, \
            'values': [ float(f) for f in list(vec) ] \
        })
    post_data = {'vectors':vectors, 'namespace':'' }
    if DEBUG: print(post_data)
    json_post_data = json.dumps(post_data)
    #print(json_post_data)
    #pinecone_url =  "https://example-hybrid-index-ee95edb.svc.us-west1-gcp.pinecone.io/vectors/upsert"
    pinecone_url =  "https://hybrid-index-2-59df0f1.svc.us-west1-gcp.pinecone.io/vectors/upsert"

    if DEBUG: return  # DEBUG=True is essentially a dry-run

    start_t = time.time()
    res = None
    try:
        r = requests.post(\
            url=pinecone_url,\
            headers=headers,\
            data=json_post_data,\
            timeout=None)
        res = r.status_code
    except:
        traceback.print_exc()
        res = "exception"
    end_t = time.time()
     
    result = {'idx':idx, 'result':res, 'timing': end_t - start_t, 'pack':PACK }
    print(result)
    results.append( result )

    df = pd.DataFrame(results)
    df.to_csv( "results.csv" )
    #print("Wrote results.csv...") 

for idx in range(0, MAX_DOCS, PACK):

    print("Processing", idx, PACK)
    dense_vectors = []
    all_tokens = []
    for i in range(PACK):
        doc = next(docs_iter)
        if DEBUG: print("DOC=", doc)
        dense_vectors.append( doc['vector'] )
        all_tokens.append( tokenizer.tokenize( doc['_text_'].lower() ) )

    if idx<SKIP_TO:
        print("skipping %d (%d)" % (idx, SKIP_TO))
        continue

    #print(type(dense_vector), dense_vector.dtype, dense_vector.shape)
    upsert(idx, dense_vectors, all_tokens)

    idx += 1
