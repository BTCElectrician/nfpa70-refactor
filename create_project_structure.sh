mkdir -p data_processing azure_search azure_function
touch main.py requirements.txt
touch data_processing/pdf_extractor.py data_processing/text_chunker.py
touch azure_search/index_creator.py azure_search/data_indexer.py
touch azure_function/function_app.py 