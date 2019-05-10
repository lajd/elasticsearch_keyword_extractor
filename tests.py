#!/usr/bin/env python3
from extractor import Extractor
from es_utils import ESUtility

import nltk
from nltk.corpus import brown
from elasticsearch import Elasticsearch, helpers
from pymongo import MongoClient


nltk.download('brown')

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

# Clients
es = Elasticsearch([{'host': 'localhost', 'port': 9200}], timeout=5, request_timeout=5)
mongo = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)


def test_es_connection():
    assert es.ping()


def test_mongo_connection():
    assert mongo[DB_NAME].command('ping')


def test_es_indexing():
    """
    Validate es-indexing
    """
    # Re-index
    es_utility = reindex()

    # Define extractor & extract keyterms, reindexing into ES
    extractor = Extractor(INDEX_NAME, FIELD_NAME, FIELD_NAME)
    extractor.extract_and_index_updates()
    # Check that the new fields exist in ES and are not empty
    updated_doc_counter = 0
    for batch in es_utility.scroll_indexed_data():
        for d in batch:
            doc = d['_source']
            # Since we're re-indexing into ES, we have to check if the document is supposed
            # to have bene udpated or not
            if FIELD_NAME in doc:
                # The document contains the keyword_extraction_field, and should be updated. Ensure that
                # the document contains updated fields (and these fields are not empty)
                assert REQUIRED_FIELDS.issubset(set(doc.keys()))
                for f in REQUIRED_FIELDS:
                    assert len(doc[f]) > 0
                updated_doc_counter += 1
            else:
                # If keyword_extraction_field is not in the document, the document should not be updated.
                # Make sure the document doesn't contain updated fields.
                assert len(REQUIRED_FIELDS.intersection(set(doc.keys()))) == 0

    assert updated_doc_counter == NUM_TEST_EXAMPLES, 'Incorrect number of samples'


def test_mongo_indexing():
    """
    Validate mongo-indexing
    """
    # Re-index
    es_utility = reindex()

    mongo.drop_database(DB_NAME)
    # Define extractor & extract keyterms, reindexing into ES
    extractor = Extractor(INDEX_NAME, FIELD_NAME, FIELD_NAME, write_updates_to_mongo=True,
                          mongo_db_name=DB_NAME, mongo_collection_name=COLL_NAME)
    # Documents should be inserted into a mongo db named 'test_db', into collection 'test_coll'
    extractor.extract_and_index_updates()
    coll = mongo[DB_NAME].get_collection(COLL_NAME)
    updated_doc_counter = 0
    for doc in coll.find():
        assert REQUIRED_FIELDS.issubset(set(doc.keys()))
        for f in REQUIRED_FIELDS:
            assert len(doc[f]) > 0
        # Since we're indexing into mongo, we and we only index updates, we don't need to check
        # if the document was supposed to be updates (all documents are updates)
        updated_doc_counter += 1
    assert updated_doc_counter == NUM_TEST_EXAMPLES, 'Incorrect number of samples'
    mongo.drop_database(DB_NAME)


class ExampleTextIterator:
    """ Example data to index """
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


def index_text_data(es, field_name, index_name, text_iterator, bsize=100, dummy_id_start=0):
    """
    Args:
        es (Elasticsearch client object): ES client
        field_name (str): Name of field to index text under
        index_name (str): Name of ES index to put data in
        text_iterator (iterable): iterable of text to index
        bsize (int): write batch size
        dummy_id_start (int):
    """
    id_counter = dummy_id_start
    chunk = []
    for text in text_iterator:
        chunk.append({"_op_type": "index", "_index": index_name, "_type": "_doc", field_name: text, '_id': id_counter})
        id_counter += 1
        if id_counter % bsize == 0:
            helpers.bulk(es, chunk, index=index_name, doc_type="_doc", refresh=True)
            chunk = []
    if len(chunk) > 0:
        helpers.bulk(es, chunk, index=index_name, doc_type="_doc", refresh=True)
    print('Finished indexing {} documents'.format(id_counter))


def reindex():
    # Re-index
    if es.indices.exists(INDEX_NAME): es.indices.delete(INDEX_NAME)
    es.indices.create(INDEX_NAME, {})
    es_utility = ESUtility(index_name=INDEX_NAME)
    # Index arbitrary amount of data under a field we're not interested in
    arbitrary_count = 100
    index_text_data(es_utility.es, 'not_used_field', INDEX_NAME, ExampleTextIterator(arbitrary_count))
    # Index NUM_TEST_EXAMPLES data under the field we are interested in
    index_text_data(es_utility.es, FIELD_NAME, INDEX_NAME, ExampleTextIterator(NUM_TEST_EXAMPLES), dummy_id_start=arbitrary_count)
    index_doc_count = es.indices.stats()['indices'][INDEX_NAME]['total']['docs']['count']
    assert index_doc_count == arbitrary_count + NUM_TEST_EXAMPLES
    return es_utility
