#!/usr/bin/env python3
import time
import string
from itertools import chain
import re

from segtok import tokenizer
from extraction.extractor import Extractor
from extraction.db_utils import ESUtility

import nltk
from nltk.corpus import brown
from elasticsearch import Elasticsearch, helpers
from pymongo import MongoClient


nltk.download('brown')

# Define indexing parameters
# ES
INDEX_NAME = 'test'
FIELD_NAME = 'data'
TOKENIZED_FIELD_NAME = 'data.tokenized'

# Mongo
DB_NAME = 'test_db'
COLL_NAME = 'test_coll'

# Test Params
NUM_TEST_EXAMPLES = 246

ELASTICSEARCH_URL = 'http://localhost:9200'
MONGO_URL = 'mongodb://localhost:27017'

# Clients
es = Elasticsearch(ELASTICSEARCH_URL, timeout=5, request_timeout=5)
mongo = MongoClient(MONGO_URL, serverSelectionTimeoutMS=2000)


# Notes:
# In all the tests we enforce min_bg_count=1, since we are using a small ES index and can't be sure
# of the count of tokens.

def test_es_connection():
    assert es.ping()


def test_mongo_connection():
    assert mongo[DB_NAME].command('ping')


def test_es_indexing_w_contexts():
    """
    Validate es-indexing
    """
    # Index both dummy-documents (not contained the desired field) and
    # documents containing the desired field
    es_utility = reindex()
    # Define extractor & extract keyterms, reindexing into ES
    extractor = Extractor(INDEX_NAME, FIELD_NAME, FIELD_NAME, extract_contexts=True, min_bg_count=1)
    extractor.extract_and_index_updates()
    time.sleep(3)  # Sleep to allow updates to index
    # Iterate though all documents. If the document is supposed to be updated
    # (i.e. FIELD_NAME is in source), then validate that the document is updated.
    updated_doc_counter = 0
    for batch in es_utility.scroll_indexed_data():
        for d in batch:
            source = d['_source']
            if FIELD_NAME in source:
                validate_keywords_contexts_offsets(source, extract_contexts=True)
                updated_doc_counter += 1
            else:
                assert len({'keyterms', 'offsets', 'contexts'}.intersection(set(source.keys()))) == 0
    # Check that the number of updated documents is as expected
    assert updated_doc_counter == NUM_TEST_EXAMPLES


def test_es_indexing_w_contexts_and_termvectors():
    """
    Validate es-indexing with termvectors
    """
    es_utility = reindex(use_termvectors=True)
    # Extract termvectors
    extractor = Extractor(INDEX_NAME, TOKENIZED_FIELD_NAME, TOKENIZED_FIELD_NAME,
                          extract_contexts=True, use_termvectors=True, min_bg_count=1)
    extractor.extract_and_index_updates()
    time.sleep(3)  # Sleep to allow updates to index
    updated_doc_counter = 0
    for batch in es_utility.scroll_indexed_data():
        for d in batch:
            source = d['_source']
            if FIELD_NAME in source:
                updated_doc_counter += 1
                termvectors = get_es_termvectors(d['_id'], TOKENIZED_FIELD_NAME)
                # Make sure the keyterms and contexts indeed come from termvectors
                field_termvec_tokens_set = set(termvectors['terms'].keys())
                keyterms = source['keyterms']
                if keyterms is None:
                    # Keyterms did not meet filter criteria (i.e. min_bg_count)
                    continue
                assert set(keyterms).issubset(field_termvec_tokens_set)
                assert set(chain(*source['contexts'])).issubset(field_termvec_tokens_set)
            else:
                assert len({'keyterms', 'offsets', 'contexts'}.intersection(set(source.keys()))) == 0
    assert updated_doc_counter == NUM_TEST_EXAMPLES


def test_es_indexing_without_contexts():
    """
    Validate es-indexing without contexts
    """
    es_utility = reindex()
    extractor = Extractor(INDEX_NAME, FIELD_NAME, FIELD_NAME, min_bg_count=1)
    extractor.extract_and_index_updates()
    time.sleep(3)  # Sleep to allow updates to index
    updated_doc_counter = 0
    for batch in es_utility.scroll_indexed_data():
        for d in batch:
            source = d['_source']
            if FIELD_NAME in source:
                validate_keywords_contexts_offsets(source, extract_contexts=False)
                updated_doc_counter += 1
            else:
                assert len({'keyterms', 'offsets', 'contexts'}.intersection(set(source.keys()))) == 0
    assert updated_doc_counter == NUM_TEST_EXAMPLES


def test_mongo_indexing_w_contexts():
    """
    Validate mongo-indexing
    """
    reindex()
    mongo.drop_database(DB_NAME)
    # Write updates to mongo instead of ES
    extractor = Extractor(
        INDEX_NAME, FIELD_NAME, FIELD_NAME, write_updates_to_mongo=True,
        mongo_db_name=DB_NAME,  mongo_collection_name=COLL_NAME, extract_contexts=True, min_bg_count=1
    )
    # Need to wait for all mongo updates to index
    extractor.extract_and_index_updates()
    time.sleep(3)  # Sleep to allow updates to index
    coll = mongo[DB_NAME].get_collection(COLL_NAME)
    updated_doc_counter = 0
    for source in coll.find():
        validate_keywords_contexts_offsets(source, index_type='mongo', extract_contexts=True)
        updated_doc_counter += 1
    assert updated_doc_counter == NUM_TEST_EXAMPLES
    mongo.drop_database(DB_NAME)


def validate_keywords_contexts_offsets(source, index_type='es', extract_contexts=False):
    """ Validate an updated document """
    assert 'keyterms' in source and 'offsets' in source
    # Keyterms are stored as a json string
    keyterms = source['keyterms']
    offsets = source['offsets']
    if keyterms is None:
        # When keyterms is None it is because the extracted keyterms could not meet the filter criteria
        # (i.e. min_bg_count). This should only happen when min_bg_count > 1
        return
    if index_type == 'es':
        field_text = source[FIELD_NAME]
    else:
        # Mongo doesn't have the raw text field; get from ES
        doc_id = source['_id']
        es_source = get_es_source(doc_id)
        field_text = es_source[FIELD_NAME]

    # Because of ambiguity with punctuation when extracting keyterms, we
    # include tokens split by punct and removed punct
    field_tokens_set = set(tokenizer.word_tokenizer(field_text))
    field_tokens_set = expand_tokens_set_to_split_by_punct(field_tokens_set)
    # To account for how ES handles punctuation, we use both tokenized and non-tokenized forms of each token
    for k in keyterms:
        try:
            assert k in field_tokens_set
        except AssertionError:
            # Try the keyterm without punct
            assert k.translate(str.maketrans('', '', string.punctuation)) in field_tokens_set

    # Check that the keyterm offsets correspond with the raw text field
    for keyterm, offset in zip(keyterms, offsets):
        for start_idx, end_idx in offset:
            assert field_text[start_idx:end_idx] == keyterm

    if extract_contexts:
        assert 'contexts' in source
        contexts = source['contexts']
        # Check that the contexts cover all the identified keyterms
        context_tokens_set = set()
        for ctx in contexts:
            context_tokens_set = context_tokens_set.union(tokenizer.word_tokenizer(ctx))
            context_tokens_set = expand_tokens_set_to_split_by_punct(context_tokens_set)
        for keyterm in keyterms:
            try:
                assert keyterm in context_tokens_set
            except AssertionError:
                # Try the keyterm without punct
                assert keyterm.translate(str.maketrans('', '', string.punctuation)) in context_tokens_set


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
                return


def index_text_data(es, field_name, index_name, text_iterator, bsize=100, num_docs_indexed=0):
    """
    Args:
        es (Elasticsearch client object): ES client
        field_name (str): Name of field to index text under
        index_name (str): Name of ES index to put data in
        text_iterator (iterable): iterable of text to index
        bsize (int): write batch size
        num_docs_indexed (int): Counter of how many docs have already been indexed
    """
    id_counter = num_docs_indexed
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


def reindex(use_termvectors=False, num_dummy_documents=100):
    # Re-index
    if es.indices.exists(INDEX_NAME): es.indices.delete(INDEX_NAME)

    if use_termvectors:
        mapping = get_tokenized_field_mapping(FIELD_NAME)
        es.indices.create(index=INDEX_NAME, body=mapping)
        es.indices.refresh(index=INDEX_NAME)
    else:
        es.indices.create(INDEX_NAME, {})
    es_utility = ESUtility(elasticsearch_url=ELASTICSEARCH_URL, index_name=INDEX_NAME)
    # Index arbitrary amount of data under a field we're not interested in
    index_text_data(es_utility.es, 'not_used_field', INDEX_NAME, ExampleTextIterator(num_dummy_documents))
    # Index NUM_TEST_EXAMPLES data under the field we are interested in
    index_text_data(es_utility.es, FIELD_NAME, INDEX_NAME, ExampleTextIterator(NUM_TEST_EXAMPLES), num_docs_indexed=num_dummy_documents)
    index_doc_count = es.indices.stats()['indices'][INDEX_NAME]['total']['docs']['count']
    assert index_doc_count == num_dummy_documents + NUM_TEST_EXAMPLES
    return es_utility


def get_es_source(doc_id):
    query = {"query": {"ids": {"values": [doc_id] }}}
    res = es.search(index=INDEX_NAME,body=query)
    return res['hits']['hits'][0]['_source']


def get_es_termvectors(doc_id, field_name):
    termvecs = es.termvectors(id=doc_id, index=INDEX_NAME, doc_type='_doc')
    return termvecs['term_vectors'][field_name]


def expand_tokens_set_to_split_by_punct(tokens_set):
    # Include words split by punctuation
    punct_split_tokens_set = set(list(chain(*[re.findall(r"[\w]+|[.,!?;\"\']", i) for i in tokens_set])))
    removed_punct_tokens_set = set([i.translate(str.maketrans('', '', string.punctuation)) for i in tokens_set])
    return tokens_set.union(punct_split_tokens_set).union(removed_punct_tokens_set)


def get_tokenized_field_mapping(field_name):
    """ Apply dummy analysis to demonstrate termvectors """
    body = {
        "settings": {
            "analysis": {
                "filter": {
                    "filter_stemmer": {
                        "type": "stemmer",
                        "language": "english"
                    }
                },
                "analyzer": {
                    "text_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "standard",
                            "lowercase",
                            "filter_stemmer"
                        ],
                    }
                }
            }
        },
        "mappings": {
            "_doc": {
                "properties": {
                    field_name: {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "tokenized": {
                                "analyzer": "text_analyzer",
                                "type": "text",
                                "term_vector": "with_positions_offsets",
                                "store": "true"
                            }
                        }
                    }
                }
            }
        }
    }
    return body
