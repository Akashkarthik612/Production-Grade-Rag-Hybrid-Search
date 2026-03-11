from src.rag.data.toy_docs import toy_documents
from src.rag.ingestion.chunking import chunk_text


def test_chunking(toy_documents, chunk_text):
    '''will test the chnuking of the paragraphs'''
    all_chunks = []
    for doc in toy_documents:
        doc_id = doc["id"]
        doc_text = doc["text"]
        
        chunk_texts = chunk_text(doc_text, max_tokens=500, overlap = 50) #oad the chunck function here to chunk the text
        for i, ct in enumerate(chunk_texts):
            chunk_id = f"{doc_id}_chunk_{i}"
            all_chunks.append({"id": chunk_id, "text": ct})
            print(f"Chunk ID: {chunk_id}, Chunk Text: {ct[:200]}...")  # Print the first 100 characters of the chunk for verification
    return all_chunks


x = test_chunking(toy_documents, chunk_text)
print(f"Total chunks generated: {len(x)}")