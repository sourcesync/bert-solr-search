import bz2
import time
import requests
import json
import traceback
import pandas as pd
import sys
import os
import dotenv
from collections import Counter
from dotenv import dotenv_values
from data_utils import parse_dbpedia_data, SearchEngine, EmbeddingModel, enrich_doc_with_vectors
from transformers import AutoTokenizer
from transformers import BertTokenizerFast

# Load env vars
config = dotenv_values(".env") 

# Determine if we also send tokens
HYBRID = True

# Set to True to print a bunch of debugging stuff
DEBUG = False

# The compressed text corpus
input_file = '../long_abstracts_en.ttl.bz2'

# Number of vectors to pack into a request payload
PACK    =  25

# Change this constant to vary the number of indexed abstracts
# set to -1 to index all
MAX_DOCS = 1000000

# Figure out if we are skipping past uploaded payloads
SKIP_TO = 0
if os.path.exists("results.csv.last"):
    df = pd.read_csv( "results.csv.last" )
    SKIP_TO = df['idx'].max()
    print("SKIP_TO=", SKIP_TO)

# Read the input compressed file as is, without decompressing.
# Though disks are cheap, isn't it great to save them?
print("Reading BZ2 file")
source_file = bz2.BZ2File(input_file, "r")

# Get a document iterator.
print("Computing vectors from scratch")
docs_iter = parse_dbpedia_data(source_file, MAX_DOCS)
docs_iter = enrich_doc_with_vectors(docs_iter, EmbeddingModel.HUGGING_FACE_SENTENCE, SearchEngine.ELASTICSEARCH)

# Create a tokenizer
#tokenizer = AutoTokenizer.from_pretrained('transfo-xl-wt103')
tokenizer = BertTokenizerFast.from_pretrained( 'bert-base-uncased')

results = []

def build_dict(input_batch):
    # store a batch of sparse embeddings
    sparse_emb = []

    # iterate through input batch
    for token_ids in input_batch:
        # convert the input_ids list to a dictionary of key to frequency values
        d = dict(Counter(token_ids))
        # remove special tokens and append sparse vectors to sparse_emb list
        sparse_emb.append({ key: d[key] for key in d if key not in [101, 102, 103, 0] })

    # return sparse_emb list
    return sparse_emb

def generate_sparse_vectors(context_batch):
    if DEBUG: print("CONTEXT BATCH", context_batch)

    # create batch of input_ids
    inputs = tokenizer(
           context_batch, padding=True,
           truncation=True,
           max_length=1024
    )['input_ids']
    if DEBUG: print("inputs", inputs)

    # create sparse dictionaries
    sparse_embeds = build_dict([inputs])
    if DEBUG: print("sparse embeds", sparse_embeds)
    return sparse_embeds

def upsert(idx, vecs, toks, sparses):
    # create POST headers
    headers = { \
        'Content-Type':'application/json', \
        'Api-Key':config["API_KEY"]
    }
    
    # create the per-item payload
    vectors = []
    for i, vec in enumerate(vecs):
        idint = idx+i
        vectors.append( { \
            'id': str(idint), \
            'metadata': {'context':toks[i]}, \
            'sparse_values': sparses[i][0], \
            'values': [ float(f) for f in list(vec) ] \
        })

    # jsonify the entire POST payload
    post_data = {'vectors':vectors, 'namespace':'' }
    json_post_data = json.dumps(post_data)
    if DEBUG: print("POST DATA", post_data)

    pinecone_url =  "https://hybrid-index-example-ee95edb.svc.us-west1-gcp.pinecone.io/hybrid/vectors/upsert"
    if DEBUG: 
        print("PINECONE URL", pinecone_url)
        return  # DEBUG=True is essentially a dry-run

    # perform the POST call
    start_t = time.time()
    res = None
    try:
        r = requests.post(\
            url=pinecone_url,\
            headers=headers,\
            data=json_post_data,\
            timeout=None)
        res = str(r.status_code) + " - " + r.text
        if (r.status_code!=200):
            print("ERR: web service call returned", r.status_code)
    except:
        traceback.print_exc()
        res = "exception"
        print("ERR: web service call generated exception:", res)
    end_t = time.time()
    
    # create a results object for this function
    result = {'idx':idx, 'result':res, 'timing': end_t - start_t, 'pack':PACK }
    print(result)

    # append to global results list and write all to a CSV
    results.append( result )
    df = pd.DataFrame(results)
    df.to_csv( "results.csv" )
    if DEBUG: print("Wrote results.csv...") 

for idx in range(0, MAX_DOCS, PACK):
    print("Processing", idx, PACK)

    # Create vectors and tokens from the DOC(s)
    dense_vectors = []
    all_tokens = []
    sparse_vectors = []
    for i in range(PACK):
        doc = next(docs_iter)
        if DEBUG: print("DOC=", doc)
        dense_vectors.append( doc['vector'] )
        toks = tokenizer.tokenize( doc['_text_'].lower() ) 
        all_tokens.append( toks )
        sparse_vec = generate_sparse_vectors( doc['_text_'] )
        sparse_vectors.append( sparse_vec )

    # some logic to skip items already upserted
    if idx<SKIP_TO:
        print("skipping %d (%d)" % (idx, SKIP_TO))
        continue

    # do the upsert
    upsert(idx, dense_vectors, all_tokens, sparse_vectors)

    idx += 1
