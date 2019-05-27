import re
from itertools import chain
from collections import defaultdict

from elasticsearch_dsl import Search


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
        "stored_fields": [],
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


def extract_keyterms_from_queries(queries_batch, es_client, stopwords, min_bg_count):
    keyterms_batch = []
    resp = es_client.msearch(body=queries_batch)
    results = [r['aggregations']['sample']['keywords']['buckets'] for r in resp['responses']]
    for r in results:
        keyterms = set([i['key'] for i in r if (i['key'] not in stopwords and i['bg_count'] >= min_bg_count)])
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
        "stored_fields": [],
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


def extract_contexts_and_keyterm_offsets(resp, field_name, leftsep='<em>', rightsep='</em>', extract_context=False):
    batch_toks_to_locs = []
    batch_contexts = []
    # Iterate through offset and highlight responses-pairs
    for i in range(0, len(resp) - 1, 2):
        # Get response dicts
        offsets_resp = resp[i].to_dict()
        highlights_resp = resp[i + 1].to_dict()
        # Extract contexts and offsets
        _offset = offsets_resp['hits']['hits']
        _highlight = highlights_resp['hits']['hits']
        try:
            term_offsets = list(chain(*[parse_offsets(line) for line in _offset[0]['highlight'][field_name]]))
        except KeyError:
            # When `highlight` is not in the offset, this means that we couldn't find a keyterm meeting
            # the criteria (eg. min_bg_count was not achieved for any found keyterm)
            # All we can do in this situation is return empty results
            batch_contexts.append(None)
            batch_toks_to_locs.append(None)
            continue
        # Extract the keyterms and contexts from the highlighted field
        keyterms = []
        doc_contexts = []
        for line in _highlight[0]['highlight'][field_name]:
            keyterms.extend(re.findall(r'{}(.*?){}'.format(leftsep, rightsep), line))
            if extract_context:
                doc_contexts.append(re.sub(r'{}|{}'.format(leftsep, rightsep), '', line))

        # Create a mapping from tokens to offsets
        doc_toks_to_locs = defaultdict(list)
        for i, tok in enumerate(keyterms):
            doc_toks_to_locs[tok].append(term_offsets[i])

        batch_toks_to_locs.append(doc_toks_to_locs)
        if extract_context:
            batch_contexts.append(doc_contexts)

    if extract_context:
        return batch_toks_to_locs, batch_contexts
    else:
        return batch_toks_to_locs, None


def extract_termvectors_contexts_and_keyterm_offsets(doc_termvectors, keyterms_set, word_fragment_size=3, extract_context=False):
    contexts = []
    doc_tokens = list(doc_termvectors.keys())

    # Extract the positions and offsets of the keyterms
    positions = []
    keyterm_offsets = defaultdict(list)
    for k, x in doc_termvectors.items():
        if k in keyterms_set:
            for token_locs in x['tokens']:
                positions.append(token_locs['position'])
                keyterm_offsets[k].append((token_locs['start_offset'], token_locs['end_offset']))
    if len(positions) == 0:
        # No keyterms could be extracted because keyterms did not meet filter criteria (i.e. min_bg_count)
        return None, None

    if extract_context:
        # In order to find context windows, we use the sorted positions.
        sorted_positions = sorted(positions)
        # Iterate through the positions and determine the context windows
        start_pos = max(sorted_positions[0] - word_fragment_size, 0)
        end_pos = sorted_positions[0] + word_fragment_size
        for pos in sorted_positions[1:]:
            if pos - word_fragment_size > end_pos:
                # Start of a new context window; append the previous window and start another
                contexts.append(doc_tokens[start_pos: end_pos])
                start_pos = pos - word_fragment_size
                end_pos = pos + word_fragment_size
            else:
                # Part of the previous context window; update the end pos
                end_pos = pos + word_fragment_size
        contexts.append(doc_tokens[start_pos: end_pos])
        return keyterm_offsets, contexts
    else:
        return keyterm_offsets, None


def parse_offsets(line):
    """ Parse offsets returned from search-highlighter plugin """
    roi = line.split(':')[1].split(':')[0].split(',')
    _term_offsets = [(int(x[0]), int(x[1])) for x in [r.split('-') for r in roi]]
    return _term_offsets
