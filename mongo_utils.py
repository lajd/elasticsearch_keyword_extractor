#!/usr/bin/env python3
from threading import Thread

from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError


class MongoUtility:
    """
    Write-utility for mongoDB
    """
    def __init__(self, db_name, collection_name, do_upsert=True):
        # Parameters
        self.db_name = db_name
        self.collection_name = collection_name
        self.upsert = do_upsert
        # Client/collection
        client = MongoClient('mongodb://localhost:27017/')
        assert client[db_name].command('ping'), 'Cannot ping mongo client on localhost:27017'
        print('Successfully pinged mongo')
        self.collection = client[db_name].get_collection(collection_name)
        # Write threads
        self.threads = []

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
            print(bwe.details)

    def n_running_threads(self):
        return sum([t.isAlive() for t in self.threads])

