#!/usr/bin/env python3
import re
from itertools import chain

from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search


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
    
    def scroll_indexed_data(self, bsize=128, only_ids=False, randomized=False,
                            fields_must_exist=[], fields_must_not_exist=[]):
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
            yield(data_scroll['hits']['hits'])
            data_scroll = self.es.scroll(scroll_id=sid, scroll='5m')
            # Update the scroll ID
            sid = data_scroll['_scroll_id']
            # Get the number of results that returned in the last scroll
            scroll_size = len(data_scroll['hits']['hits'])
    
    def text_field_iterator(self, field_name, tokenizer=None):
        """
        Args:
            field_name (str): Text field name to extract tokens for
            tokenizer (function): Tokenizer which takes as input a string
                and returns an iterable of string tokens. The default tokenizer
                is a whitespace tokenizer
        """
        if tokenizer is None: tokenizer = lambda x: x.split()
        # Do a full pass over the data set yielding the tokens from the provided
        # text field
        for batch in self.scroll_indexed_data():
            for d in batch:
                source = d['_source']
                text = source[field_name]
                tokens = tokenizer(text)
                yield tokens

    def format_update_for_es(self, doc_update):
        return {'_op_type': 'update', '_index': self.index_name,
                '_type': '_doc', 'doc': doc_update['body'], '_id': doc_update['_id']}

    def update_documents(self, updates, total_count):
        res = helpers.bulk(self.es, updates)
        print("Total indexed: {} into ES; current res: {}".format(total_count, res))
        return res


def get_keyterms_query(_id, field_name, dg, shard_size=100):
    """ Get keyterms query by looking for significant text in a single document

    min_doc_count must be 1, since only a single _id is being passed.

    Args:
        field_name (str): field name to extract keyterms from
        dg (int): max number of keyterms to extract
        shard_size (int): shard_size parameter for significant text. Set to -1
            to use the full index (memory/time scales with shard size).
    """
    query = {
        "size": 0,
        "query": {
            "ids": {
                "values": _id
            }
        },
        "aggregations": {
            "sample": {
                "sampler": {
                    "shard_size": shard_size
                },
                "aggregations": {
                    "keywords": {
                        "significant_text": {
                            "field": field_name,
                            "size": dg,
                            'min_doc_count': 1,
                            'filter_duplicate_text': True,
                        }
                    }
                }
            }
        }
    }
    return query


def extract_keyterms_from_queries(queries_batch, es_client, stopwords):
    keyterms_batch = []
    resp = es_client.msearch(body=queries_batch)
    results = [r['aggregations']['sample']['keywords']['buckets'] for r in resp['responses']]
    for r in results:
        keyterms = set([i['key'] for i in r if i['key'] not in stopwords])
        keyterms_batch.append(list(keyterms))
    return keyterms_batch


def get_context_query(ids, keyterms, es_client, field_name):
    highlight_template = get_highlight_template(ids, keyterms, field_name, offsets=False)
    offset_template = get_highlight_template(ids, keyterms, field_name, offsets=True)
    highlight_search = Search().using(es_client).update_from_dict(highlight_template)
    offsets_search = Search().using(es_client).update_from_dict(offset_template)
    return offsets_search, highlight_search


def get_highlight_template(ids, keyterms, field_name, offsets=False, max_fragments=8, fragment_size=100):
    _should_match = [{'match': {field_name: f}} for f in keyterms]
    template = {
        'query': {
            'bool': {
                'filter': {'ids': {'values': ids}},
                'should': _should_match
            },
        },
        "highlight": {
            "order": 'none',
            'number_of_fragments': max_fragments,
            'fragment_size': fragment_size,
            "fields": {
                field_name: {
                    "type": "experimental",
                    "options": {
                        "return_offsets": offsets,
                        'remove_high_freq_terms_from_common_terms': False,
                    },
                },
            },
        }
    }
    return template


def extract_keyterm_offsets_contexts(offsets, highlights, field_name, leftsep='<em>', rightsep='</em>'):
    _offset = offsets['hits']['hits']
    _highlight = highlights['hits']['hits']
    assert len(_offset) == len(_highlight) == 1
    term_offsets = list(chain(*[parse_offsets(line) for line in _offset[0]['highlight'][field_name]]))
    keyterms = []
    contexts = []
    for line in _highlight[0]['highlight'][field_name]:
        keyterms.extend(re.findall(r'{}(.*?){}'.format(leftsep, rightsep), line))
        clean_context = re.sub(r'{}|{}'.format(leftsep, rightsep), '', line)
        contexts.append(clean_context)
    return keyterms, contexts, term_offsets


def extract_contexts_from_queries(resp, field_name):
    batch_keyterms = []
    batch_contexts = []
    batch_offsets = []
    # Iterate through offset and highlight responses-pairs
    for i in range(0, len(resp) - 1, 2):
        # Get response dicts
        offsets_resp = resp[i].to_dict()
        highlights_resp = resp[i + 1].to_dict()
        # Extract contexts and offsets
        keyterms, contexts, offsets = extract_keyterm_offsets_contexts(
            offsets_resp, highlights_resp, field_name
        )
        #
        batch_keyterms.append(keyterms)
        batch_contexts.append(contexts)
        batch_offsets.append(offsets)
    return batch_keyterms, batch_contexts, batch_offsets


def parse_offsets(line):
    """ Parse offsets returned from search-highlighter plugin """
    roi = line.split(':')[1].split(':')[0].split(',')
    _term_offsets = [(int(x[0]), int(x[1])) for x in [r.split('-') for r in roi]]
    return _term_offsets
