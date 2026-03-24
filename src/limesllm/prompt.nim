## prompt.nim -- Context + query -> prompt template assembly.

{.experimental: "strict_funcs".}

import std/strutils

# =====================================================================================================================
# Types
# =====================================================================================================================

type
  PromptTemplate* = object
    system*: string
    context_prefix*: string
    context_suffix*: string
    query_prefix*: string

# =====================================================================================================================
# Templates
# =====================================================================================================================

proc default_template*(): PromptTemplate =
  PromptTemplate(
    system: "You are a helpful assistant. Answer the question based on the provided context. If the context does not contain enough information, say so.",
    context_prefix: "Context:\n",
    context_suffix: "\n",
    query_prefix: "Question: "
  )

proc code_template*(): PromptTemplate =
  PromptTemplate(
    system: "You are a code assistant. Answer the question using only the provided source code context. Be precise and reference specific functions or types.",
    context_prefix: "Source code:\n```\n",
    context_suffix: "\n```\n",
    query_prefix: "Question: "
  )

# =====================================================================================================================
# Assembly
# =====================================================================================================================

proc assemble_prompt*(context: string, query: string,
                      tmpl: PromptTemplate = default_template()): string =
  ## Assemble a full prompt from context, query, and template.
  var parts: seq[string]
  if tmpl.system.len > 0:
    parts.add(tmpl.system)
  parts.add(tmpl.context_prefix & context & tmpl.context_suffix)
  parts.add(tmpl.query_prefix & query)
  parts.join("\n\n")

proc assemble_chat_prompt*(context: string, query: string,
                           tmpl: PromptTemplate = default_template()
                          ): tuple[system: string, user: string] =
  ## Assemble system and user messages for chat-style APIs.
  let user_msg = tmpl.context_prefix & context & tmpl.context_suffix & "\n" & tmpl.query_prefix & query
  (system: tmpl.system, user: user_msg)

proc estimate_tokens*(text: string): int =
  ## Rough token estimate: word count / 0.75.
  let words = text.splitWhitespace()
  int(float(words.len) / 0.75)
