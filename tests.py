#!/usr/bin/env python3
import string
import time
from collections import deque

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
NUM_TEST_EXAMPLES = 246
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
    # TODO: Better way of waiting for all updates to finish?
    time.sleep(5)
    # Check that the new fields exist in ES and are not empty
    updated_doc_counter = 0
    for batch in es_utility.scroll_indexed_data():
        for d in batch:
            source = d['_source']
            # Since we're re-indexing into ES, we have to check if the document is supposed
            # to have bene udpated or not
            if FIELD_NAME in source:
                # The document contains the keyword_extraction_field, and should be updated. Ensure that
                # the document contains updated fields (and these fields are not empty)
                validate_keywords_contexts_offsets(source)
                assert REQUIRED_FIELDS.issubset(set(source.keys()))
                for f in REQUIRED_FIELDS:
                    assert len(source[f]) > 0
                updated_doc_counter += 1
            else:
                # If keyword_extraction_field is not in the document, the document should not be updated.
                # Make sure the document doesn't contain updated fields.
                assert len(REQUIRED_FIELDS.intersection(set(source.keys()))) == 0

    assert updated_doc_counter == NUM_TEST_EXAMPLES, 'Incorrect number of samples'


def test_mongo_indexing():
    """
    Validate mongo-indexing
    """
    # Re-index
    reindex()
    mongo.drop_database(DB_NAME)
    # Define extractor & extract keyterms, reindexing into ES
    extractor = Extractor(INDEX_NAME, FIELD_NAME, FIELD_NAME, write_updates_to_mongo=True,
                          mongo_db_name=DB_NAME, mongo_collection_name=COLL_NAME)
    # Documents should be inserted into a mongo db named 'test_db', into collection 'test_coll'
    extractor.extract_and_index_updates()
    # TODO: Better way of waiting for all updates to finish?
    time.sleep(5)
    coll = mongo[DB_NAME].get_collection(COLL_NAME)
    updated_doc_counter = 0
    for source in coll.find():
        validate_keywords_contexts_offsets(source, index_type='mongo')
        assert REQUIRED_FIELDS.issubset(set(source.keys()))
        for f in REQUIRED_FIELDS:
            assert len(source[f]) > 0
        # Since we're indexing into mongo, we and we only index updates, we don't need to check
        # if the document was supposed to be updates (all documents are updates)
        updated_doc_counter += 1
    assert updated_doc_counter == NUM_TEST_EXAMPLES, 'Incorrect number of samples'
    mongo.drop_database(DB_NAME)


def validate_keywords_contexts_offsets(source, index_type='es'):
    """
    Iterate through the collection and validate keyword/contexts/offsets
    """
    assert index_type in {'es', 'mongo'}, 'index type must be `es` or `mongo`'
    keyterms, contexts, offsets = source['keyterms'], source['contexts'], source['offsets']
    if index_type == 'es':
        data = source[FIELD_NAME]
        # Only the ES index has the raw data
        # Keywords should come in the order they appear in the document
        data_tokens, data_keyterms = deque(data.split()), deque(keyterms)
        ref_tok = data_keyterms.popleft()
        while data_tokens:
            compare_tok = data_tokens.popleft()
            if compare_tok == ref_tok:
                continue
            elif compare_tok.translate(str.maketrans('', '', string.punctuation)) == ref_tok:
                continue
            if len(data_keyterms) == 0: break
            ref_tok = data_keyterms.popleft()
        # If data_keyterms is not 0, then the keyterms didn't come in order (or weren't found in the document)
        assert len(data_keyterms) == 0
        # Check that the term identified by the offset is indeed the keyterm found
        for k, o in zip(keyterms, offsets):
            start_idx, end_idx = o
            assert data[start_idx:end_idx] == k
    # Check that keyterms come in order in the contexts
    # In general the elements of context can be overlapping string, and so we
    # don't expect context_data_toks to be the same as data_tokens. The ordering
    # of keyterms should be the same though.
    context_data_toks, data_keyterms = deque(' '.join(contexts).split(' ')), deque(keyterms)
    ref_tok = data_keyterms.popleft()
    while context_data_toks:
        compare_tok = context_data_toks.popleft()
        if compare_tok == ref_tok:
            continue
        elif compare_tok.translate(str.maketrans('', '', string.punctuation)) == ref_tok:
            continue
        if len(data_keyterms) == 0: break
        ref_tok = data_keyterms.popleft()
    assert len(data_keyterms) == 0


class ExampleTextIterator:
    """ Example data to index """
    def __init__(self, n_examples, n_sentences_accuum=5):
        self.n_examples = n_examples
        self.n_sentences_accuum = n_sentences_accuum

    def __iter__(self):
        total_sents = self.n_examples * self.n_sentences_accuum
        sents_generator = brown.sents()

        total_count = 0
        mini_batch_count = 0
        sent = ''
        for sent_toks in sents_generator:
            sent += ' '.join(sent_toks)
            mini_batch_count += 1
            if mini_batch_count == self.n_sentences_accuum:
                yield sent
                sent = ''
                mini_batch_count = 0
            total_count += 1
            if total_count == total_sents:
                raise StopIteration


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
