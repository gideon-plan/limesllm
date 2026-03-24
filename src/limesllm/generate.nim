## generate.nim -- Prompt -> llama inference -> response.
##
## Abstracts over llama API: tokenize, decode, sample loop.

{.experimental: "strict_funcs".}

import lattice

# =====================================================================================================================
# Types
# =====================================================================================================================

type
  GenerateConfig* = object
    max_tokens*: int       ## Max tokens to generate
    temperature*: float32
    top_k*: int32
    top_p*: float32

  GenerateFn* = proc(prompt: string, config: GenerateConfig): Result[string, RagError] {.raises: [].}
    ## Function that generates text from a prompt.
    ## Abstracts over llama tokenize + decode + sample loop.

# =====================================================================================================================
# Configuration
# =====================================================================================================================

proc default_generate_config*(): GenerateConfig =
  GenerateConfig(max_tokens: 512, temperature: 0.7, top_k: 40, top_p: 0.95)

# =====================================================================================================================
# Generation
# =====================================================================================================================

proc generate*(prompt: string, gen_fn: GenerateFn,
               config: GenerateConfig = default_generate_config()
              ): Result[string, RagError] =
  ## Generate a response from the given prompt.
  gen_fn(prompt, config)

proc generate_with_context*(context: string, query: string,
                            gen_fn: GenerateFn,
                            config: GenerateConfig = default_generate_config()
                           ): Result[string, RagError] =
  ## Convenience: assemble prompt inline and generate.
  let prompt = "Context:\n" & context & "\n\nQuestion: " & query
  gen_fn(prompt, config)
