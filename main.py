#!/usr/bin/env python3
import argparse

from extraction.extractor import Extractor


parser = argparse.ArgumentParser(
    description='Extract keyterms, contexts and keyterm-offsets from ES and update into either ES or Mongo\n'
                'Example usage: python main.py <es_index_name> <es_keyword_field> <es_highlight_field> [--options]'
)
# Positional required arguments
parser.add_argument('es_index_name', type=str, help='Name of elasticsearch index')
parser.add_argument('es_keyword_field', type=str, help='Name of elasticsearch field to extract keywords from')
parser.add_argument('es_highlight_field', type=str, help='Name of elasticsearch field to extract highlights from')
# Optional arguments
parser.add_argument('--keywords_field_name', type=str, required=False, default='keywords', help='Extracted keywords field name')
parser.add_argument('--offsets_field_name', type=str, required=False, default='offsets', help='Extracted offsets field name')
parser.add_argument('--contexts_field_name', type=str, required=False, default='contexts', help='Extracted contexts field name')
parser.add_argument('--shard_size', type=int, required=False, default=100, help='Shard size for significant text')
parser.add_argument('--must_have_fields', type=str, required=False, nargs='*', default=[],
                    help='Fields which must be contained in the document to be elligible for keyword extraction')
parser.add_argument('--must_not_have_fields', type=str, required=False, nargs='*', default=[],
                    help='Fields which must not be contained in the document to be elligible for keyword extraction')
parser.add_argument('--use_termvectors', type=bool, default=False, required=False,
                    help='Whether to extract offsets using termvectors. Use if using an anlyzer.')
parser.add_argument('--termvectors_window_size', type=int, default=3, required=False,
                    help='The context-window size to use to the left and right of a keyterm')
parser.add_argument('--extract_contexts', type=bool, default=False, required=False,
                    help='Whether to extract and index contexts')
parser.add_argument('--max_doc_keyterms', type=int, default=16, required=False, help='Maximum numebr of keyterms to extract')
parser.add_argument('--bsize', type=int, default=128, required=False, help='Read/write batch size')
parser.add_argument('--max_doc_qsize', type=int, default=256, help='Maximum number of documents to be queued; limits memory usage')
parser.add_argument('--min_bg_count', type=int, default=1,
                    help='Minimum number of occurences of the term in the index to be elligible as a keyword. Defaults to the minimum of 1')
parser.add_argument('--n_extracting_threads', type=int, default=4,
                    help='Number of threads to use for extracting contexts and offsets')
parser.add_argument('--n_indexing_threads', type=int, default=4,
                    help='Number of threads to use for writing updates into mongo/ES')
parser.add_argument('--elasticsearch_url', type=str, default='http://localhost:9200', required=False,
                    help='Elasticsearch URL -- defaults to http://localhost:9200')
# Optional arguments (writing updates to mongo)
parser.add_argument('--write_updates_to_mongo', type=bool, default=False, required=False,
                    help='Whether to index updates to MongoDB rather than elasticsearch. '
                         'Used when MongoDB is the primary data store')
parser.add_argument('--mongo_db_name', type=str, default=None, required=False,
                    help='When `write_updates_to_mongo = True`, the MongoDB database name to write to')
parser.add_argument('--mongo_collection_name', type=str, default=None, required=False,
                    help='When `write_updates_to_mongo = True`, the MongoDB collection name to write to')
parser.add_argument('--mongo_url', type=str, default='mongodb://localhost:27017/', required=False,
                    help='Mongo URL -- defaults to mongodb://localhost:27017/')


if __name__ == '__main__':
    args = parser.parse_args()
    kwargs = vars(args)
    print('-------------Running keyword extraction with args-------------')
    print(kwargs)
    extractor = Extractor(**kwargs)
    extractor.extract_and_index_updates()
