## embed.nim -- Text chunking + embedding pipeline.
##
## Chunks text into segments, generates embeddings via llama, stores in limes.

{.experimental: "strict_funcs".}

import std/[strutils, tables, math]
import lattice

# =====================================================================================================================
# Types
# =====================================================================================================================

type
  Chunk* = object
    text*: string
    index*: int
    source*: string
    metadata*: Table[string, string]

  ChunkConfig* = object
    max_tokens*: int       ## Max tokens per chunk
    overlap_tokens*: int   ## Overlap between consecutive chunks

  EmbedResult* = object
    chunk*: Chunk
    embedding*: seq[float32]

# =====================================================================================================================
# Chunking
# =====================================================================================================================

proc default_chunk_config*(): ChunkConfig =
  ChunkConfig(max_tokens: 512, overlap_tokens: 64)

proc chunk_text*(text: string, source: string = "",
                 config: ChunkConfig = default_chunk_config()): seq[Chunk] =
  ## Split text into overlapping chunks by paragraph boundaries.
  ## Falls back to sentence splitting, then hard token-count splitting.
  let paragraphs = text.split("\n\n")
  var current = ""
  var idx = 0
  for para in paragraphs:
    let trimmed = para.strip()
    if trimmed.len == 0:
      continue
    # Estimate tokens as words / 0.75 (rough approximation)
    let combined = if current.len > 0: current & "\n\n" & trimmed else: trimmed
    let est_tokens = int(ceil(float(combined.split(' ').len) / 0.75))
    if est_tokens > config.max_tokens and current.len > 0:
      result.add(Chunk(text: current, index: idx, source: source))
      inc idx
      # Overlap: keep last portion of current
      let words = current.split(' ')
      let overlap_words = min(config.overlap_tokens, words.len)
      if overlap_words > 0:
        current = words[words.len - overlap_words ..< words.len].join(" ") & "\n\n" & trimmed
      else:
        current = trimmed
    else:
      current = combined
  if current.strip().len > 0:
    result.add(Chunk(text: current, index: idx, source: source))

proc chunk_documents*(docs: seq[(string, string)],
                      config: ChunkConfig = default_chunk_config()): seq[Chunk] =
  ## Chunk multiple (source, text) pairs.
  for (source, text) in docs:
    result.add(chunk_text(text, source, config))

# =====================================================================================================================
# Embedding generation
# =====================================================================================================================

type
  EmbedFn* = proc(text: string): Result[seq[float32], RagError] {.raises: [].}
    ## Function that generates an embedding vector from text.
    ## Abstracts over llama get_embeddings or any other embedding backend.

proc embed_chunks*(chunks: seq[Chunk], embed_fn: EmbedFn): Result[seq[EmbedResult], RagError] =
  ## Generate embeddings for a sequence of chunks.
  var results: seq[EmbedResult]
  for chunk in chunks:
    let emb = embed_fn(chunk.text)
    if emb.is_bad:
      return Result[seq[EmbedResult], RagError].bad(emb.err)
    results.add(EmbedResult(chunk: chunk, embedding: emb.val))
  Result[seq[EmbedResult], RagError].good(results)
