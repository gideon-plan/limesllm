## tlimesllm.nim -- Tests for limesllm RAG pipeline.

{.experimental: "strict_funcs".}

import std/[unittest, tables, strutils]
import limesllm

# =====================================================================================================================
# Chunking tests
# =====================================================================================================================

suite "embed: chunking":
  test "chunk single short text":
    let chunks = chunk_text("Hello world.", "test.txt")
    check chunks.len == 1
    check chunks[0].text == "Hello world."
    check chunks[0].source == "test.txt"
    check chunks[0].index == 0

  test "chunk multi-paragraph text":
    let text = "First paragraph with some words.\n\nSecond paragraph with more words.\n\nThird paragraph here."
    let config = ChunkConfig(max_tokens: 10, overlap_tokens: 2)
    let chunks = chunk_text(text, "doc.md", config)
    check chunks.len >= 2
    for c in chunks:
      check c.source == "doc.md"

  test "chunk empty text":
    let chunks = chunk_text("", "empty.txt")
    check chunks.len == 0

  test "chunk_documents multiple docs":
    let docs = @[("a.txt", "Content A."), ("b.txt", "Content B.")]
    let chunks = chunk_documents(docs)
    check chunks.len == 2
    check chunks[0].source == "a.txt"
    check chunks[1].source == "b.txt"

# =====================================================================================================================
# Embed tests
# =====================================================================================================================

suite "embed: embedding":
  test "embed_chunks with mock embedder":
    let mock_embed: EmbedFn = proc(text: string): Result[seq[float32], RagError] {.raises: [].} =
      Result[seq[float32], RagError].good(@[1.0'f32, 2.0'f32, 3.0'f32])
    let chunks = @[Chunk(text: "hello", index: 0, source: "test")]
    let result = embed_chunks(chunks, mock_embed)
    check result.is_good
    check result.val.len == 1
    check result.val[0].embedding == @[1.0'f32, 2.0'f32, 3.0'f32]

  test "embed_chunks propagates error":
    let fail_embed: EmbedFn = proc(text: string): Result[seq[float32], RagError] {.raises: [].} =
      Result[seq[float32], RagError].bad(RagError(msg: "model error"))
    let chunks = @[Chunk(text: "hello", index: 0, source: "test")]
    let result = embed_chunks(chunks, fail_embed)
    check result.is_bad

# =====================================================================================================================
# Retrieve tests
# =====================================================================================================================

suite "retrieve":
  test "retrieve filters by min_score":
    let mock_embed_q: EmbedQueryFn = proc(text: string): Result[seq[float32], RagError] {.raises: [].} =
      Result[seq[float32], RagError].good(@[1.0'f32])
    let mock_query: QueryFn = proc(qe: seq[float32], k: int): Result[seq[RetrievedChunk], RagError] {.raises: [].} =
      Result[seq[RetrievedChunk], RagError].good(@[
        RetrievedChunk(text: "high", score: 0.9, source: "a"),
        RetrievedChunk(text: "low", score: 0.1, source: "b")])
    let config = RetrieveConfig(top_k: 5, min_score: 0.5, max_context_chars: 4096)
    let result = retrieve("query", mock_embed_q, mock_query, config)
    check result.is_good
    check result.val.len == 1
    check result.val[0].text == "high"

  test "assemble_context joins chunks":
    let chunks = @[
      RetrievedChunk(text: "chunk1", score: 0.9, source: "a"),
      RetrievedChunk(text: "chunk2", score: 0.8, source: "b")]
    let ctx = assemble_context(chunks, " | ")
    check ctx == "chunk1 | chunk2"

# =====================================================================================================================
# Prompt tests
# =====================================================================================================================

suite "prompt":
  test "assemble_prompt default template":
    let result = assemble_prompt("some context", "what is this?")
    check result.contains("Context:")
    check result.contains("some context")
    check result.contains("Question: what is this?")

  test "assemble_chat_prompt":
    let (system, user) = assemble_chat_prompt("ctx", "query?")
    check system.len > 0
    check user.contains("ctx")
    check user.contains("query?")

  test "estimate_tokens":
    check estimate_tokens("one two three four") >= 4

# =====================================================================================================================
# Generate tests
# =====================================================================================================================

suite "generate":
  test "generate with mock":
    let mock_gen: GenerateFn = proc(p: string, c: GenerateConfig): Result[string, RagError] {.raises: [].} =
      Result[string, RagError].good("Generated answer: " & p[0..5])
    let result = generate("What is Nim?", mock_gen)
    check result.is_good
    check result.val.contains("Generated")

# =====================================================================================================================
# RAG pipeline tests
# =====================================================================================================================

suite "rag pipeline":
  test "query end-to-end with mocks":
    let mock_embed: EmbedFn = proc(text: string): Result[seq[float32], RagError] {.raises: [].} =
      Result[seq[float32], RagError].good(@[1.0'f32])
    let mock_embed_q: EmbedQueryFn = proc(text: string): Result[seq[float32], RagError] {.raises: [].} =
      Result[seq[float32], RagError].good(@[1.0'f32])
    let mock_query: QueryFn = proc(qe: seq[float32], k: int): Result[seq[RetrievedChunk], RagError] {.raises: [].} =
      Result[seq[RetrievedChunk], RagError].good(@[
        RetrievedChunk(text: "Nim is a systems language.", score: 0.95, source: "nim.md")])
    let mock_gen: GenerateFn = proc(p: string, c: GenerateConfig): Result[string, RagError] {.raises: [].} =
      Result[string, RagError].good("Nim is a compiled language.")
    let pipeline = new_rag_pipeline(mock_embed, mock_embed_q, mock_query, mock_gen)
    let result = pipeline.query("What is Nim?")
    check result.is_good
    check result.val == "Nim is a compiled language."

  test "ingest with mocks":
    let mock_embed: EmbedFn = proc(text: string): Result[seq[float32], RagError] {.raises: [].} =
      Result[seq[float32], RagError].good(@[1.0'f32])
    let mock_embed_q: EmbedQueryFn = proc(text: string): Result[seq[float32], RagError] {.raises: [].} =
      Result[seq[float32], RagError].good(@[1.0'f32])
    let mock_query: QueryFn = proc(qe: seq[float32], k: int): Result[seq[RetrievedChunk], RagError] {.raises: [].} =
      Result[seq[RetrievedChunk], RagError].good(@[])
    let mock_gen: GenerateFn = proc(p: string, c: GenerateConfig): Result[string, RagError] {.raises: [].} =
      Result[string, RagError].good("")
    let mock_store: StoreFn = proc(e: seq[float32], t: string, m: Table[string, string]): Result[string, RagError] {.raises: [].} =
      Result[string, RagError].good("id_1")
    let pipeline = new_rag_pipeline(mock_embed, mock_embed_q, mock_query, mock_gen)
    let result = pipeline.ingest("Some document text.", "doc.txt", mock_store)
    check result.is_good
    check result.val >= 1
