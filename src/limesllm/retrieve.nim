## retrieve.nim -- Query vector store, rank results, assemble context window.

{.experimental: "strict_funcs".}

import std/[algorithm, strutils, tables]
import lattice

# =====================================================================================================================
# Types
# =====================================================================================================================

type
  RetrievedChunk* = object
    text*: string
    score*: float32
    source*: string
    metadata*: Table[string, string]

  RetrieveConfig* = object
    top_k*: int            ## Number of results to retrieve
    min_score*: float32    ## Minimum similarity score
    max_context_chars*: int ## Max total characters in assembled context

  QueryFn* = proc(query_embedding: seq[float32], top_k: int): Result[seq[RetrievedChunk], RagError] {.raises: [].}
    ## Function that queries the vector store.
    ## Abstracts over limes query or any other backend.

  EmbedQueryFn* = proc(text: string): Result[seq[float32], RagError] {.raises: [].}
    ## Function that embeds a query string.

# =====================================================================================================================
# Configuration
# =====================================================================================================================

proc default_retrieve_config*(): RetrieveConfig =
  RetrieveConfig(top_k: 5, min_score: 0.0, max_context_chars: 4096)

# =====================================================================================================================
# Retrieval
# =====================================================================================================================

proc retrieve*(query: string, embed_query_fn: EmbedQueryFn, query_fn: QueryFn,
               config: RetrieveConfig = default_retrieve_config()
              ): Result[seq[RetrievedChunk], RagError] =
  ## Embed query, search vector store, filter and rank results.
  let qemb = embed_query_fn(query)
  if qemb.is_bad:
    return Result[seq[RetrievedChunk], RagError].bad(qemb.err)
  let raw = query_fn(qemb.val, config.top_k)
  if raw.is_bad:
    return Result[seq[RetrievedChunk], RagError].bad(raw.err)
  # Filter by min_score and sort by descending score
  var filtered: seq[RetrievedChunk]
  for r in raw.val:
    if r.score >= config.min_score:
      filtered.add(r)
  filtered.sort(proc(a, b: RetrievedChunk): int =
    if a.score > b.score: -1
    elif a.score < b.score: 1
    else: 0)
  # Trim to max context chars
  var total_chars = 0
  var trimmed: seq[RetrievedChunk]
  for r in filtered:
    if total_chars + r.text.len > config.max_context_chars:
      break
    trimmed.add(r)
    total_chars += r.text.len
  Result[seq[RetrievedChunk], RagError].good(trimmed)

proc assemble_context*(chunks: seq[RetrievedChunk], separator: string = "\n\n---\n\n"): string =
  ## Join retrieved chunks into a single context string.
  var parts: seq[string]
  for c in chunks:
    parts.add(c.text)
  parts.join(separator)
