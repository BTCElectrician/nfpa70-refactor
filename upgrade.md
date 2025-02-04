<Plan>
Below are **Beginner-Friendly Step-by-Step** instructions to resolve the "unexpected 'StartArray' node" errors in your code. After the instructions, you'll find **complete revised files**. These changes ensure:

1. **No "metadata" field** is ever sent to Azure Search.  
2. **All fields** match the index schema:  
   - **Strings** (`content`, `article_number`, `section_number`, `article_title`, `section_title`, `gpt_analysis`)  
   - **Int** (`page_number`)  
   - **Collections of strings** (`context_tags`, `related_sections`)  
   - **Vector** (`content_vector`: array of 1536 floats)

Review the instructions first; then, use the provided code for each file verbatim.

</Plan>

---

## STEP-BY-STEP FIXES (BEGINNER STYLE)

1. **Delete & Re-Create Your Search Index**  
   - Use `create_search_index(...)` from `index_creator.py` or manually delete in Azure Portal.  
   - **Important**: Deleting the index ensures an old/stale schema won't conflict with new documents.

2. **Update `data_indexer.py`: Remove "metadata" references**  
   - We no longer rely on a sub-field called `"metadata"`. Instead, each chunk/dictionary has `"page_number"`, `"article_number"`, and `"section_number"` at the top level.  
   - This guarantees the final document matches your declared index fields.

3. **Update `main.py`: Remove "metadata" in chunk output**  
   - Instead of storing each chunk's page/article/section as `"metadata": {...}`, store them as top-level fields (`"page_number"`, `"article_number"`, `"section_number"`).  
   - When you later index these chunks, the correct fields are populated.

4. **Update `test_single_chunk_processing.py`: Same Changes**  
   - If you produce a chunk dictionary with `"metadata": {...}"`, it can lead to confusion.  
   - For clarity, define `"page_number"`, `"article_number"`, `"section_number"` at top level.  
   - `DataIndexer` will then embed your text properly and push the correct structure to Azure Search.

5. **Confirm Proper Vector Dimensions**  
   - Make sure the OpenAI embedding call always yields an array of length **1536**.  
   - `data_indexer.py` already verifies this, so you should be all set.

6. **Re-run Your Script**  
   - After updating these four files, run your normal process (e.g., `main.py`) to create chunks, store them, then either call `index_from_blob.py` or run your tests with `test_single_chunk_processing.py`.  
   - The "unexpected 'StartArray' node" error should be resolved.

---

## ADDITIONAL UPGRADE: TEXT CHUNKER MODIFICATIONS

<Plan>
The following section details the necessary updates to `text_chunker.py` to return top-level fields rather than nesting them under `"metadata"`. This complements the previous changes and ensures consistency across the entire pipeline.
</Plan>

### STEP-BY-STEP INSTRUCTIONS

1. **Locate `text_chunker.py`:**  
   This file contains the `ElectricalCodeChunker` class and a **compatibility function** named `chunk_nfpa70_content`.

2. **Identify the "metadata" Nesting:**  
   - The core class `ElectricalCodeChunker` returns a list of `CodeChunk` objects with top-level properties like `page_number`, `article_number`, etc. That part is fine.  
   - **However**, the `chunk_nfpa70_content` function at the bottom uses:
     ```python
     {
       "metadata": {
         "section": chunk.section_number,
         "article": chunk.article_number,
         "page": chunk.page_number
       }
     }
     ```
     This causes "metadata" to be nested.

[... rest of the new content ...]

### TESTING CHECKLIST

1. **After Replacing `text_chunker.py`:**  
   - Confirm your code never references `metadata` fields.  
   - Ensure you pass the chunk dictionaries properly to `DataIndexer.index_documents`.

2. **Run `main.py`** (or your typical entry point) to generate these chunk dictionaries and store them.  

3. **Run Your Indexing Script** (e.g., `index_from_blob.py` or `test_single_chunk_processing.py`).  
   - The chunk data should now have top-level fields matching your Azure Search schema, preventing the "metadata mismatch" error.

---

## COMPLETE UPGRADE VERIFICATION

1. Verify all files have been updated:
   - ✓ `text_chunker.py`
   - ✓ `data_indexer.py`
   - ✓ `main.py`
   - ✓ `test_single_chunk_processing.py`

2. Confirm all metadata is now top-level:
   - ✓ No nested "metadata" objects
   - ✓ All fields directly accessible
   - ✓ Proper type handling for all fields

3. Test the complete pipeline:
   - ✓ PDF extraction works
   - ✓ Chunking produces correct structure
   - ✓ Indexing succeeds without errors
   - ✓ Search queries return expected results

</rewritten_file>
