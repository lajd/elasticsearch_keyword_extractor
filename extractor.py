#!/usr/bin/env python3
import time
from threading import Thread
from collections import deque

from es_utils import (
    ESUtility,
    get_keyterms_query,
    extract_keyterms_from_queries,
    get_context_query,
    extract_contexts_from_queries,
)
from mongo_utils import MongoUtility
from elasticsearch_dsl import MultiSearch
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


class TaskCompletionManager:
    def __init__(self, fetching_target, populating_target, indexing_target):
        self.state_map = {
            'fetching': {'completed': 0, 'target': fetching_target},
            'populating': {'completed': 0, 'target': populating_target},
            'indexing': {'completed': 0, 'target': indexing_target},
        }

    def task_running(self, task):
        return self.state_map[task]['completed'] != self.state_map[task]['target']

    def add_completed(self, task):
        self.state_map[task]['completed'] += 1


class Extractor:
    def __init__(self, es_index_name, es_keyword_field, es_highlight_field,
                 shard_size=100, must_have_fields=[], must_not_have_fields=[],
                 keyword_stopwords='en', max_doc_keyterms=16, bsize=128, max_doc_qsize=256,
                 write_updates_to_mongo=False, mongo_db_name=None, mongo_collection_name=None,
                 n_extracting_threads=4, n_indexing_threads=4):
        """
        Args:
            es_index_name (str): Name of elasticsearch index
            es_keyword_field (str): Name of field in ES from which to extract significant text
            es_highlight_field (str): Name of field in ES from which to highlight and obtain contexts for keywords
            shard_size (int): Shard size for significant text
            must_have_fields (list): Fields which must be in the document (in addition to keyword and highlight fields)
            must_not_have_fields (list): Fields which must not be in the document to be elligible for keyword extraction
            keyword_stopwords (str): Set of tokens which are not allowed to be keywords
            max_doc_keyterms (int): Maximum number of keyterms to consider for each document
            bsize (int): Read batch size when pulling from ES
            max_doc_qsize (int): Maximum queue size for read/write threads; smaller limits memory
                but may take longer
            write_updates_to_mongo (bool): Whether to write updates to mongo instead of ES
            mongo_db_name (str): If inserting updates into mongo instead of ES, the database name to use
            mongo_collection_name (str):  If inserting updates into mongo instead of ES, the collection name to use
            n_extracting_threads (int): Number of threads for extracting keyterms, contexts and offsets
            n_indexing_threads (int): Number of threads for inserting (keyterms, contexts, offsets) updates into ES/mongo
        """
        if keyword_stopwords is None:
            keyword_stopwords = set()
        elif keyword_stopwords == 'en':
            keyword_stopwords = set(stopwords.words('english'))
        elif not isinstance(keyword_stopwords, set):
            raise RuntimeError('Keywords stopwords must be a set')
        # Parameters
        self.es_index_name = es_index_name
        self.es_keyword_field = es_keyword_field
        self.es_highlight_field = es_highlight_field
        self.shard_size = shard_size
        self.keyword_stopwords = keyword_stopwords
        self.max_doc_qsize = max_doc_qsize
        self.max_doc_keyterms = max_doc_keyterms
        self.bsize = bsize
        self.must_have_fields = must_have_fields
        self.must_not_have_fields = must_not_have_fields
        # Elasticsearch/mongo connections
        self.es_utility = ESUtility(es_index_name, read_bsize=bsize)
        self.update_formatter, self.write_updater = self.get_writer_and_formatter(
            write_updates_to_mongo, mongo_db_name, mongo_collection_name
        )
        # Accumulators
        self.keyterms_query_batches = deque()
        self.updates = deque()
        self.indexed_doc_counter = 0
        # Helpers
        self.task_manager = TaskCompletionManager(1, n_extracting_threads, n_indexing_threads)

    def get_writer_and_formatter(self, write_updates_to_mongo, mongo_db_name, mongo_collection_name):
        if write_updates_to_mongo and mongo_db_name is not None and mongo_collection_name is not None:
            # Instead of writing updates to ES, write to mongo
            mongo_utility = MongoUtility(mongo_db_name, mongo_collection_name)
            update_formatter = mongo_utility.format_update_for_mongo
            write_updater = mongo_utility.update_documents
            print('Updates will be written to mongo instead of ES')
        else:
            update_formatter = self.es_utility.format_update_for_es
            write_updater = self.es_utility.update_documents
            print('Updates will be written to ES')
        return update_formatter, write_updater

    def fetch_keyterm_queries(self):
        """ Get keyterm-extraction queries for each document
        
        Task 1: Scroll through ES, fetching document ids. Form a keyterms query 
        for each document id and add to keyterms_query_batch. Execute on a single
        threads since it's not a bottleneck.
        """
        batch = {'_ids': [], 'keyterms': []}
        batch_keyterms_queries = []
        # We only consider documents which have the keyword and highlight field, plus any additional fields specified
        _must_have_fields = self.must_have_fields + [self.es_keyword_field, self.es_highlight_field]
        ids_generator = self.es_utility.scroll_indexed_data(bsize=self.bsize, only_ids=True,
                    fields_must_exist=_must_have_fields, fields_must_not_exist=self.must_not_have_fields)
        
        for i, doc_batch in enumerate(ids_generator, 1):
            # If the queue is full, wait
            while len(self.keyterms_query_batches) * self.bsize > self.max_doc_qsize:
                time.sleep(1)
            # Extract queries for the batch
            for doc in doc_batch:
                _id = doc['_id']
                batch['_ids'].append(_id)
                keyterms_queries = [
                    {"index": self.es_index_name},
                    get_keyterms_query(_id, self.es_keyword_field, self.max_doc_keyterms, self.shard_size)
                ]
                batch_keyterms_queries.extend(keyterms_queries)
            # Add the batch to the queue
            batch['keyterms'].extend(extract_keyterms_from_queries(
                batch_keyterms_queries, self.es_utility.es, self.keyword_stopwords
            ))
            self.keyterms_query_batches.append(batch)
            batch = {'_ids': [], 'keyterms': []}
            batch_keyterms_queries = []
        # Process remaining queries
        if len(batch_keyterms_queries) > 0:
            batch['keyterms'].extend(extract_keyterms_from_queries(
                batch_keyterms_queries, self.es_utility.es, self.keyword_stopwords
            ))
            self.keyterms_query_batches.append(batch)
        # Signify the end of keyterm extraction by appending the finished flag
        self.task_manager.add_completed('fetching')

    def extract_keyterms_and_contexts(self):
        """ Execute keyterms and highlight/offset queries

        Task 2: Extract (keyterms, contexts, offsets) updates and
        add to queue for indexing. Execute task over multiple threads
        to speed things up.
        """
        while self.task_manager.task_running('fetching') or len(self.keyterms_query_batches) > 0:
            try:
                # Get the next batch
                batch = self.keyterms_query_batches.popleft()
            except IndexError:
                # Queue is empty
                time.sleep(1)
                continue

            # Execute keyterms queries
            ms = MultiSearch(index=self.es_index_name).using(self.es_utility.es)
            for _id, _keyterms in zip(batch['_ids'], batch['keyterms']):
                offsets_search, highlight_search = get_context_query(
                    [_id], _keyterms, self.es_utility.es, self.es_highlight_field
                )
                ms = ms.add(offsets_search).add(highlight_search)
            #
            resp = ms.execute(raise_on_error=True)
            batch_keyterms, batch_contexts, batch_offsets = extract_contexts_from_queries(resp, self.es_highlight_field)
            #
            for _id, keyterms, contexts, offsets in zip(batch['_ids'], batch_keyterms, batch_contexts, batch_offsets):
                _update = {
                    '_id': _id,
                    'body': {
                        "keyterms": keyterms,
                        'contexts': contexts,
                        'offsets': offsets
                    }
                }
                self.updates.append(_update)
        # Notify that the thread has finished
        self.task_manager.add_completed('populating')

    def index_updates(self):
        """ Index (keyterms, contexts, offsets) updates

        Task 3: Index udpates into either ES or Mongo. Execute over
        multiple threads.
        """
        counter = 0
        updates = []
        while self.task_manager.task_running('populating') or len(self.updates) > 0:
            try:
                doc = self.updates.popleft()
            except IndexError:
                time.sleep(1)
                continue

            updates.append(self.update_formatter(doc))
            counter += 1
            if counter % self.bsize == 0:
                self.indexed_doc_counter += len(updates)
                self.write_updater(updates, self.indexed_doc_counter)
                updates = []

        # Write remaining updates
        if len(updates) > 0:
            self.indexed_doc_counter += len(updates)
            self.write_updater(updates, self.indexed_doc_counter)
        # Extracted keyterms queue is empty and no longer being populated
        self.task_manager.add_completed('indexing')

    def extract_and_index_updates(self):
        """ Run all tasks """
        # Queue keyterms queries
        t1 = time.time()
        Thread(target=self.fetch_keyterm_queries).start()

        # Extract embeddings
        for _ in range(self.task_manager.state_map['populating']['target']):
            Thread(target=self.extract_keyterms_and_contexts).start()

        # Index updates
        for _ in range(self.task_manager.state_map['indexing']['target']):
            Thread(target=self.index_updates).start()

        while self.task_manager.task_running('indexing'):
            time.sleep(1)

        t2 = time.time()
        print('Finished indexing {} updates in {}s'.format(self.indexed_doc_counter, round(t2 - t1, 2)))
