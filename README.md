# elasticsearch_keyword_extractor
- Script for extracting (keywords, contexts and keyword-offsets) and re-indexing (keywords, contexts and keyword-offsets) into either ES or MongoDB</br>

</br>

## Requirements
1) Elasticsearch version 6.5.4 (due to plugin dependency)</br>
2) Running ES service on localhost:9200</br>
3) Running Mongo service on localhost:27017</br>
3) Installed search-highlighter plugin (https://github.com/wikimedia/search-highlighter)</br>

</br>

## Usage
Use this script if you're trying to extract keywords from a field in ElasticSearch, perhaps for a machine learning task.</br>
The script does the following: </br>
1) Extracts keywords from each indexed document using ES's significant text API</br>
2) Extract `context-windows` (ES highlight) for keyterms. This can be used, for example, to obtain context-aware embedding vectors. Also
   extracts keyterm positional-offsets (with respect to source-field text). Positional offsets use the search-highlighter plugin
   https://github.com/wikimedia/search-highlighter</br>
3) Re-indexes updates (keyterms, contexts, offsets) into either ES or mongo. Specify destination with arguments.</br>

</br>

## Examples

Use case: Read from ES index <es_index_name> and extract (keyterms, contexts, offsets) from field <es_field_name>.
Extract highlights and positional offsets from field <es_highlight_field>. </br>

</br>

To update each document in <es_index_name>: </br>

python main.py <es_index_name> <es_field_name> <es_highlight_field> </br>

</br>
To update documents in MongoDB in database <mongo_db_name> in collection <mongo_collection_name>. If the document cannot
be matched in the collection, upsert is enabled. </br>

python main.py <es_index_name> <es_field_name> <es_highlight_field> --write_updates_to_mongo True --mongo_db_name <mongo_db_name> --mongo_collection_name <mongo_collection_name> </br>

</br>

## Tests
pytest tests.py </br>