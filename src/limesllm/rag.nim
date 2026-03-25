## rag.nim -- End-to-end RAG pipeline: embed -> store -> retrieve -> generate.

{.experimental: "strict_funcs".}

import std/tables
import basis/code/choice, embed, retrieve, prompt, generate

# =====================================================================================================================
# Types
# =====================================================================================================================

type
  RagPipeline* = object
    embed_fn*: EmbedFn
    embed_query_fn*: EmbedQueryFn
    query_fn*: QueryFn
    gen_fn*: GenerateFn
    chunk_config*: ChunkConfig
    retrieve_config*: RetrieveConfig
    generate_config*: GenerateConfig
    prompt_template*: PromptTemplate

  StoreFn* = proc(embedding: seq[float32], text: string, metadata: Table[string, string]): Choice[string] {.raises: [].}
    ## Function that stores a vector + metadata. Returns vector ID.

# =====================================================================================================================
# Pipeline construction
# =====================================================================================================================

proc new_rag_pipeline*(embed_fn: EmbedFn, embed_query_fn: EmbedQueryFn,
                       query_fn: QueryFn, gen_fn: GenerateFn): RagPipeline =
  RagPipeline(
    embed_fn: embed_fn,
    embed_query_fn: embed_query_fn,
    query_fn: query_fn,
    gen_fn: gen_fn,
    chunk_config: default_chunk_config(),
    retrieve_config: default_retrieve_config(),
    generate_config: default_generate_config(),
    prompt_template: default_template()
  )

# =====================================================================================================================
# Ingest
# =====================================================================================================================

proc ingest*(pipeline: RagPipeline, text: string, source: string,
             store_fn: StoreFn): Choice[int] =
  ## Chunk text, embed chunks, store in vector DB. Returns number of chunks stored.
  let chunks = chunk_text(text, source, pipeline.chunk_config)
  let embedded = embed_chunks(chunks, pipeline.embed_fn)
  if embedded.is_bad:
    return bad[int](embedded.err)
  var stored = 0
  for er in embedded.val:
    var meta = er.chunk.metadata
    meta["source"] = er.chunk.source
    meta["chunk_index"] = $er.chunk.index
    let id = store_fn(er.embedding, er.chunk.text, meta)
    if id.is_bad:
      return bad[int](id.err)
    inc stored
  good(stored)

# =====================================================================================================================
# Query
# =====================================================================================================================

proc query*(pipeline: RagPipeline, question: string): Choice[string] =
  ## Full RAG: retrieve context, assemble prompt, generate answer.
  let retrieved = retrieve(question, pipeline.embed_query_fn, pipeline.query_fn,
                           pipeline.retrieve_config)
  if retrieved.is_bad:
    return bad[string](retrieved.err)
  let context = assemble_context(retrieved.val)
  let full_prompt = assemble_prompt(context, question, pipeline.prompt_template)
  generate(full_prompt, pipeline.gen_fn, pipeline.generate_config)
