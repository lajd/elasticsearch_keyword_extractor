#!/usr/bin/env python3
from elasticsearch import Elasticsearch, helpers


class ESUtility:
    def __init__(self, index_name, read_bsize=100):
        self.index_name = index_name
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}], timeout=60, request_timeout=60)
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

