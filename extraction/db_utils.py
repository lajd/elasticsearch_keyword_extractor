#!/usr/bin/env python3
from elasticsearch import Elasticsearch, helpers
from threading import Thread

from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError


class MongoUtility:
    """
    Write-utility for mongoDB
    """
    def __init__(self, mongo_url, db_name, collection_name, do_upsert=True):
        # Parameters
        self.db_name = db_name
        self.collection_name = collection_name
        self.upsert = do_upsert
        # Client/collection
        client = MongoClient(mongo_url)
        assert client[db_name].command('ping'), 'Cannot ping mongo client on localhost:27017'
        print('Successfully pinged mongo')
        self.collection = client[db_name].get_collection(collection_name)
        # Write threads
        self.threads = []

    def scroll_indexed_data(self, bsize=128, fields_must_exist=[], fields_must_not_exist=[]):
        """ Method only returns the ids of matching documents
        Returns a batch, since calling functions expect a batch
        """
        exists_fields = [{f: {'$exists': True}} for f in fields_must_exist]
        not_exists_fields = [{f: {'$exists': False}} for f in fields_must_not_exist]
        cursor = self.collection.find({'$and': exists_fields + not_exists_fields}, {})
        batch = []
        for i, d in enumerate(cursor, 1):
            batch.append(d)
            if i % bsize == 0:
                yield batch
                batch = []

    def format_update_for_mongo(self, doc_update):
        _update = {k: v for k, v in doc_update['body'].items()}
        return UpdateOne(
            filter={'_id': doc_update['_id']},
            update={'$set': _update},
            upsert=self.upsert,
        )

    def update_documents(self, updates, indexed_count):
        t = Thread(target=self.async_write, args=[self.collection, updates, indexed_count])
        self.threads.append(t)
        t.start()

    @staticmethod
    def async_write(coll, updates, indexed_count):
        try:
            coll.bulk_write(updates, ordered=False)
            print("Total indexed: {} into mongo".format(indexed_count))
        except BulkWriteError as bwe:
            print("ERROR in bulk write::: {}".format(bwe.details))

    def n_running_threads(self):
        return sum([t.isAlive() for t in self.threads])


class ESUtility:
    def __init__(self, elasticsearch_url, index_name, read_bsize=100):
        self.index_name = index_name
        self.es = Elasticsearch(elasticsearch_url, timeout=60, request_timeout=60)
        assert self.es.ping(), 'Cannot ping elasticsearch on localhost:9200'
        print('Successfully pinged elasticsearch')
        # Check index exists
        if not self.es.indices.exists(index=index_name):
            raise RuntimeError('Index {} not found'.format(index_name))
        self.read_bsize = read_bsize
    
    def scroll_indexed_data(self, bsize=128, only_ids=False, randomized=False, fields_must_exist=[], fields_must_not_exist=[], use_termvectors=False):
        fields_must_exist = list(set(fields_must_exist))
        fields_must_not_exist = list(set(fields_must_not_exist))
        default_body = {
            "query": {
                "bool": {
                    "must": [{"exists": {"field": f}} for f in fields_must_exist],
                    "must_not": [{"exists": {"field": f}} for f in fields_must_not_exist]
                }
            }
        }

        if randomized:
            # Return random batches of data
            body = {
                "query": {
                    "function_score": {
                        "query": default_body['query'],
                        "random_score": {}
                    }
                }
            }
        else:
            body = default_body
        
        if only_ids:
            # Return only the id field
            body["stored_fields"] = []
        
        data_scroll = self.es.search(
            index=self.index_name,
            scroll='5m',
            size=bsize,
            body=body,
        )

        sid = data_scroll['_scroll_id']
        scroll_size = len(data_scroll['hits']['hits'])

        while scroll_size > 0:
            # Make another request to yield termvectors
            hits = data_scroll['hits']['hits']
            if use_termvectors:
                documents = {'ids': [x['_id'] for x in hits]}
                termvecs = self.es.mtermvectors(
                    body=documents, doc_type='_doc', index=self.index_name,
                    term_statistics=False, field_statistics=False
                )
                yield termvecs['docs']
            else:
                yield hits
            data_scroll = self.es.scroll(scroll_id=sid, scroll='5m')
            # Update the scroll ID
            sid = data_scroll['_scroll_id']
            # Get the number of results that returned in the last scroll
            scroll_size = len(data_scroll['hits']['hits'])

    @staticmethod
    def extract_tokens_from_termvectors(d, field_name):
        # Termvectors are already tokenized; Need to sort position
        tok_loc_tuples = []
        for tok, tok_attrs in d['term_vectors'][field_name]['terms'].items():
            tok_locs_elements = tok_attrs['tokens']
            for loc_element in tok_locs_elements:
                tok_loc_tuples.append((tok, loc_element['position']))
        tokens = [i[0] for i in sorted(tok_loc_tuples, key=lambda x: x[1])]
        return tokens

    def format_update_for_es(self, doc_update):
        return {'_op_type': 'update', '_index': self.index_name,
                '_type': '_doc', 'doc': doc_update['body'], '_id': doc_update['_id']}

    def update_documents(self, updates, total_count):
        res = helpers.bulk(self.es, updates)
        print("Total indexed: {} into ES; current res: {}".format(total_count, res))
        return res

