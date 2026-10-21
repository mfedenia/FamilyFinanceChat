# Legacy vendor fork — removed

## What was here

`custom-code-vendor-fork/` held the old Open WebUI fork: replacement
`main.py`, `config.py`, and `observability.py` files, custom PDF/UI routers,
and a PDF crawler/uploader with a bookmarklet interface. About 6,500 lines
of vendor internals supported two custom features. The superseded migration
plan, `openwebui_upgrade_decoupling.plan.md.done`, was also retired from this
directory; both remain available in Git history.

## Why it was removed

The fork coupled the platform to private implementation details, required
manual merges for upgrades, and left it stuck on Open WebUI v0.6.41.
Retaining executable code and setup instructions invited a return to that design.

**Bind-mounting replacement source over Open WebUI internals is forbidden:** it recreates the fork and breaks upstream upgrades.
**The `inject-script` bookmarklet mechanism is forbidden:** frontend JavaScript injection bypasses supported extension surfaces and breaks on UI changes.

These restrictions concern vendor source overrides and injected UI code;
normal data-volume mounts are unaffected. See the
[upstream compatibility policy](../capstone_fall2026/COMPATIBILITY_POLICY.md).

## What replaced it

- An unmodified official Open WebUI image, configured in
  [docker-compose.yml](../docker-compose.yml).
- Native Knowledge Base uploads and the public REST API via
  [tools/kb_sync](../tools/kb_sync/README.md). The custom PDF crawler was retired.
- Chat metrics through the [Filter Function](../monitoring/chat_metrics_filter.py)
  and Prometheus Pushgateway, with separate monitoring containers.
- Grading extraction through the public REST API. See
  [SETUP.md](../SETUP.md) for the supported setup and [HANDOFF.md](../HANDOFF.md)
  for migration context.

## Historical recovery only

The complete fork and migration plan are preserved at commit
`12fe7a18df6c23e12fa9c2021662e21c25bb3ecb` (the parent of this cleanup).
[Browse the historical legacy directory](https://github.com/mfedenia/FamilyFinanceChat/tree/12fe7a18df6c23e12fa9c2021662e21c25bb3ecb/legacy).
To extract a copy for historical inspection outside the working tree:

```sh
git archive 12fe7a18df6c23e12fa9c2021662e21c25bb3ecb legacy/ > ../familyfinancechat-legacy.tar
```

This is an archival reference, not a supported deployment path. Do not restore
vendor overrides or injection mechanisms to the running platform.
