#!/usr/bin/env python3
from extractor import Extractor
from es_utils import ESUtility

import nltk
from nltk.corpus import brown
from elasticsearch import Elasticsearch
from pymongo import MongoClient


nltk.download('brown')

# Clients
es = Elasticsearch([{'host': 'localhost', 'port': 9200}], timeout=5, request_timeout=5)
mongo = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)

# Define indexing parameters
# ES
INDEX_NAME = 'test'
FIELD_NAME = 'data'
# Mongo
DB_NAME = 'test_db'
COLL_NAME = 'test_coll'
# Test Params
NUM_TEST_EXAMPLES = 147
REQUIRED_FIELDS = {'keyterms', 'contexts', 'offsets'}


def test_es_indexing():
    """
    Validate es-indexing
    """
    es_util = reindex()
    # Define extractor & extract keyterms, reindexing into ES
    extractor = Extractor(INDEX_NAME, FIELD_NAME, FIELD_NAME)
    extractor.extract_and_index_updates()
    # Check that the new fields exist in ES and are not empty
    counter = 0
    for batch in es_util.scroll_indexed_data():
        for d in batch:
            doc = d['_source']
            validate_doc(doc, REQUIRED_FIELDS)
            counter += 1
    assert NUM_TEST_EXAMPLES == counter, 'Incorrect number of samples'


def test_mongo_indexing():
    """
    Validate mongo-indexing
    """
    es_util = reindex()
    mongo.drop_database(DB_NAME)
    assert mongo[DB_NAME].command('ping'), 'Cannot ping mongo client on localhost:27017'
    # Define extractor & extract keyterms, reindexing into ES
    extractor = Extractor(INDEX_NAME, FIELD_NAME, FIELD_NAME, write_updates_to_mongo=True,
                          mongo_db_name=DB_NAME, mongo_collection_name=COLL_NAME)
    # Documents should be inserted into a mongo db named 'test_db', into collection 'test_coll'
    extractor.extract_and_index_updates()
    coll = mongo[DB_NAME].get_collection(COLL_NAME)
    counter = 0
    for doc in coll.find():
        validate_doc(doc, REQUIRED_FIELDS)
        counter += 1
    assert counter == NUM_TEST_EXAMPLES, 'Incorrect number of samples'
    mongo.drop_database(DB_NAME)


class ExampleTextIterator:
    def __init__(self, n_examples):
        self.n_examples = n_examples

    def __iter__(self):
        c = 0
        for sent_toks in brown.sents():
            sent = ' '.join(sent_toks)
            if c == self.n_examples:
                raise StopIteration
            yield sent
            c += 1


def reindex():
    # Delete index if it already exists
    if es.indices.exists(INDEX_NAME):
        es.indices.delete(INDEX_NAME)
    # Create the index
    es.indices.create(INDEX_NAME, {})
    # Index some data
    es_utility = ESUtility(index_name=INDEX_NAME)
    es_utility.index_text_data(FIELD_NAME, INDEX_NAME, ExampleTextIterator(NUM_TEST_EXAMPLES))
    return es_utility


def validate_doc(doc, pos_len_fields):
    assert pos_len_fields.issubset(set(doc.keys()))
    for f in pos_len_fields:
        assert len(doc[f]) > 0
