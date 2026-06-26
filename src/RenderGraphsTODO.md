# RenderGraph TODO

**Moved.** The live task list is now consolidated in [`status.md`](../status.md) (repo root) — one tracker
instead of three drifting ones. Capability audit lives in `docs/rendergraph-gap-analysis.md`.

Kept here only as the historical record of what closed:

- **Critical bugs (both fixed):** release-build OOM null-deref (P4 — always-on loud fail + safe skip);
  `SizeKind::Relative` scale-0.5 mis-render (P3 — `resolve_size` rounds before the cast).
- **High priority (closed):** `force_keep()` / explicit outputs; persistent non-temporal resources +
  `initialize(target, hash)` bake; MSAA + resolve + stencil routing; transient `create_buffer` +
  phase-4 within-frame memory aliasing.
- **Still open** (see `status.md` for the ranked backlog): resource-type validation, cycle guards
  (`resolve_size` + pass DAG), release-build validation-coverage decision, no-GPU regression check,
  `kMaxAccess`/arena grow path, pool/usage-derivation dedup, demo polish (HDR bloom, cube geometry).
