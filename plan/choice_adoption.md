# Choice/Life Adoption Plan: limesllm

## Summary

- **Error type**: `RagError` defined in lattice.nim -- move to `embed.nim`
- **Files to modify**: 5 + re-export module
- **Result sites**: 20
- **Life**: Not applicable

## Steps

1. Delete `src/limesllm/lattice.nim`
2. Move `RagError* = object of CatchableError` to `src/limesllm/embed.nim`
3. Add `requires "basis >= 0.1.0"` to nimble
4. In every file importing lattice:
   - Replace `import.*lattice` with `import basis/code/choice`
   - Replace `Result[T, E].good(v)` with `good(v)`
   - Replace `Result[T, E].bad(e[])` with `bad[T]("limesllm", e.msg)`
   - Replace `Result[T, E].bad(RagError(msg: "x"))` with `bad[T]("limesllm", "x")`
   - Replace return type `Result[T, RagError]` with `Choice[T]`
5. Update re-export: `export lattice` -> `export choice`
6. Update tests
