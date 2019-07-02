# elasticsearch_keyword_extractor
- Example script for extracting keywords, contexts and keyword-offsets from an Elasticsearch index, and re-indexing  the obtained keywords, contexts and keyword-offsets into either ES or MongoDB. </br>

</br>

## Requirements
1) Elasticsearch version 6.5.4 (due to plugin dependency, although it's easy to update the plugin)</br>
2) Running ES service on localhost:9200</br>
3) Running Mongo service on localhost:27017</br>
3) Installed search-highlighter plugin (https://github.com/wikimedia/search-highlighter)</br>

</br>

## Description
This script can be used (either as an example or out of the box) if you're trying to extract keywords from a text field in ElasticSearch. The keywords are extracted based on TFIDF values (see ES's siginificant text API for details). The script also allows for the extraction of keyword-character-ranges, as well as the surrounding word contexts. For my application, I was extracting keywords and contexts for a contextual-embedding method (eg. BERT), and using the offsets to identify word correlation for use in the MGAN model https://arxiv.org/pdf/1902.10580.pdf. </br>

The script does the following: </br>
1) Extracts keywords from each indexed document using ES's significant text API</br>
2) Extract context-windows (ES highlight) for keyterms. This can be used, for example, to obtain context-aware embedding vectors. 
3) Extracts keyterm positional-offsets (with respect to source-field text). Positional offsets use the search-highlighter plugin https://github.com/wikimedia/search-highlighter</br>
3) Re-indexes updates (keyterms, contexts, offsets) into either ES or mongo.</br>

</br>

### Example output for extracting keyterms, offsets and contexts from a raw text field
Typically, we want to both identify and highlight keyterms in the same field. Given a text field `<es_field_name> = text_data` and `es_highlight_field = text_data`</br>, the example output is: </br>

</br>

`text_data`</br>
"Pitcher Steve Barber joined the club one week ago after completing his hitch under the Army's accelerated wintertime military course , also at Fort Knox , Ky. .The 22-year-old southpaw enlisted earlier last fall than did Hansen .Baltimore's bulky spring-training contingent now gradually will be reduced as Manager Paul Richards and his coaches seek to trim it down to a more streamlined and workable unit . " Take a ride on this one " , Brooks Robinson greeted Hansen as the Bird third sacker grabbed a bat , headed for the plate and bounced a third-inning two-run double off the left-centerfield wall tonight .It was the first of two doubles by Robinson , who was in a mood to celebrate ."
 </br>

`keyterms`</br>
['Pitcher', 'completing', 'wintertime', 'southpaw', 'gradually', 'trim', 'greeted', 'Bird', 'sacker', 'grabbed', 'bat', 'double', 'wall', 'tonight', 'celebrate'] </br>

</br>

`offsets` </br>
[[[0, 7]], [[56, 66]], [[106, 116]], [[176, 184]], [[278, 287]], [[353, 357]], [[455, 462]], [[477, 481]], [[488, 494]], [[495, 502]], [[505, 508]], [[567, 573]], [[599, 603]], [[604, 611]], [[680, 689]]] </br>

</br>

`contexts`</br>
["Pitcher Steve Barber joined the club one week ago after completing his hitch under the Army's accelerated", 'accelerated wintertime military course , also at Fort Knox , Ky. .The 22-year-old southpaw enlisted earlier', 'contingent now gradually will be reduced as Manager Paul Richards and his coaches seek to trim it down to', "one '' , Brooks Robinson greeted Hansen as the Bird third sacker grabbed a bat , headed for the plate", 'bounced a third-inning two-run double off the left-centerfield wall tonight .It was the first of two doubles', ' .It was the first of two doubles by Robinson , who was in a mood to celebrate .']
</br>

</br>

The (start, end) index of keyterm `i` in `<es_field_name>` is `offset[i]`, and all keyterms appear at least once in one of the context strings. The main use for the contexts are to obtain contextual embeddings (although this is easy enough to do with just the offsets and raw text field.) </br>

</br>

### Example output for extracting keyterms, offsets and contexts from an analysed text field
Sometimes we are interested in extracting keyterms, offsets and contexts from an analyzed text field (eg. after stemming and lowering). The script supports extracting this data from an analyzed text field by using the ES termvectors API. </br>

</br>

`text_data`</br>
"Rice has not played since injuring a knee in the opener with Maryland ." He's looking a lot better , and he's able to run ", Meek explained . " We'll let him do a lot of running this week , but I don't know if he'll be able to play " .The game players saw the Air Force film Monday , ran for 30 minutes , then went in , while the reserves scrimmaged for 45 minutes . " We'll work hard Tuesday , Wednesday and Thursday " , Meek said , " and probably will have a good scrimmage Friday " ."

 </br>

`keyterms`</br>
['45', 'air', 'explain', 'film', 'good', "he'll", 'lot', 'maryland', 'meek', 'ran', 'reserv', 'rice', 'scrimmag', 'thursdai', "we'll"] </br>

</br>

`offsets` </br>
[[[358, 360]], [[264, 267]], [[133, 142]], [[274, 278]], [[467, 471]], [[213, 218]], [[89, 92], [166, 169]], [[61, 69]], [[128, 132], [427, 431]], [[288, 291]], [[334, 342]], [[0, 4]], [[343, 353], [472, 481]], [[413, 421]], [[147, 152], [373, 378]]] </br>

</br>

`contexts`</br>
[['30', '45', 'a'], ['do', "don't", 'explain', 'film', 'for', 'forc', 'fridai', 'game', 'good', 'ha'], ['have', "he'", "he'll", 'him', 'i', 'if', 'in', 'injur', 'knee', 'know', 'let', 'look', 'lot'], ['not', 'of', 'open', 'plai', 'player', 'probabl'], ['said', 'saw', 'scrimmag', 'sinc', 'the', 'then', 'thi', 'thursdai', 'to', 'tuesdai'], ['went', 'while', 'will', 'with', 'work']] </br>

</br>

The (start, end) index of keyterm `i` in `<es_field_name>` is `offset[i]`, and all keyterms appear at least once in one of the context strings. The main use for the contexts are to obtain contextual embeddings (although this is easy enough to do with just the offsets and raw text field.) </br>

</br>


## Example usage

Use case: Read from ES index <es_index_name> and extract keyterms, offsets and contexts from a raw text field <es_field_name>.
Extract contexts and offsets from field <es_highlight_field>. </br>

</br>

To update each document in <es_index_name>: </br>
</br>
`python main.py <es_index_name> <es_field_name> <es_highlight_field>` </br>

</br>
To update documents in MongoDB in database <mongo_db_name> in collection <mongo_collection_name>. If the document cannot
be matched in the collection, upsert is enabled. </br>
</br>

`python main.py <es_index_name> <es_field_name> <es_highlight_field> --write_updates_to_mongo True --mongo_db_name <mongo_db_name> --mongo_collection_name <mongo_collection_name>` </br>

Use case: Read from ES index <es_index_name> and extract keyterms, offsets and contexts from an analyzed text field <es_field_name>. </br>

</br>

`python main.py <es_index_name> <es_field_name> <es_field_name> --use_termvectors True` </br>

</br>

Again, we can index updates into either ES or MongoDB. Note that when using termvectors, the field for which we extract keyterms must be the same for which we extract offsets/contexts. </br>

</br>


## All parameters

| Positional Arguments        | Description           |
| ------------- |:-------------:|
| es_index_name      | Name of elasticsearch index |
| es_keyword_field      | Name of elasticsearch field to extract keywords from      |
| es_highlight_field | Name of elasticsearch field to extract highlights from. Must be the same as es_keyword_field when use_termvectors is set to True      |

</br>

| Optional Arguments        | Description           |
| ------------- |:-------------:|
| --shard_size      | Shard size for significant text |
| --must_have_fields       | Fields which must be contained in the document to be elligible for keyword extraction     |
| --must_not_have_fields | Fields which must not be contained in the document to be elligible for keyword extraction  |
| --use_termvectors |  Whether to extract offsets using termvectors. Use if using an anlyzer.|
| --termvectors_window_size | The context-window size to use to the left and right of a keyterm |
| --extract_contexts | Whether to extract and index contexts |
| --max_doc_keyterms | Maximum numebr of keyterms to extract |
| --bsize  | Read/write batch size |
| --max_doc_qsize | Maximum number of documents to be queued; limits memory usage |
| --min_bg_count  | Minimum number of occurences of the term in the index to be elligible as a keyword. Defaults to the minimum of 1 |
| --n_extracting_threads | Number of threads to use for extracting contexts and offsets |
| --n_indexing_threads | Number of threads to use for writing updates into mongo/ES |

</br>

| Optional arguments (writing updates to mongo)       |    Description           |
| ------------- |:-------------:|
| --write_updates_to_mongo | Whether to index updates to MongoDB rather than elasticsearch. Used when MongoDB is the primary data store |
| --mongo_db_name      | When write_updates_to_mongo = True, the MongoDB database name to write to      |
| --mongo_collection_name | When write_updates_to_mongo = True, the MongoDB collection name to write to      |

</br>

## Tests
pytest tests.py </br>
