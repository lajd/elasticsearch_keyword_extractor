#!/usr/bin/env python3
import argparse

from extractor import Extractor


parser = argparse.ArgumentParser(
    description='Extract keyterms, contexts and keyterm-offsets from ES and update into either ES or Mongo\n'
                'Example usage: python main.py <es_index_name> <es_keyword_field> <es_highlight_field> [--options]'
)
# Positional required arguments
parser.add_argument('es_index_name', type=str, help='Name of elasticsearch index')
parser.add_argument('es_keyword_field', type=str, help='Name of elasticsearch field to extract keywords from')
parser.add_argument('es_highlight_field', type=str, help='Name of elasticsearch field to extract highlights from')

# Optional arguments
parser.add_argument('--max_doc_keyterms', type=int, default=16, required=False, help='Maximum numebr of keyterms to extract')
parser.add_argument('--bsize', type=int, default=128, required=False, help='Read/write batch size')
parser.add_argument('--max_doc_qsize', type=int, default=256, help='Maximum number of documents to be queued; limits memory usage')
parser.add_argument('--n_populating_threads', type=int, default=4,
                    help='Number of threads to use for extracting contexts and offsets')
parser.add_argument('--n_indexing_threads', type=int, default=4,
                    help='Number of threads to use for writing updates into mongo/ES')
# Optional arguments (writing updates to mongo)
parser.add_argument('--write_updates_to_mongo', type=bool, default=False, required=False,
                    help='Whether to index updates to MongoDB rather than elasticsearch. '
                         'Used when MongoDB is the primary data store')
parser.add_argument('--mongo_db_name', type=str, default=None, required=False,
                    help='When `write_updates_to_mongo = True`, the MongoDB database name to write to')
parser.add_argument('--mongo_collection_name', type=str, default=None, required=False,
                    help='When `write_updates_to_mongo = True`, the MongoDB collection name to write to')


if __name__ == '__main__':
    args = parser.parse_args()
    kwargs = vars(args)
    extractor = Extractor(**kwargs)
    extractor.extract_and_index_updates()