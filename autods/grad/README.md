# Graph Rag Api Doc - GRAD

## Quick start

1. Add new repository to GRAD

```bash
uv run autods/grad/grad.py add https://github.com/owner/repo_name
```

2. Ask repository using GRAD

```bash
uv run autods/grad/grad.py ask https://github.com/owner/repo_name "How to use LightAutoML library?"
```

## Troubleshooting

### Too many open files (os error 24)

If you encounter an error like:

```
LanceError(IO): Too many open files (os error 24)
```

This is a macOS/Linux file descriptor limit issue. LanceDB (the vector database) opens many files for its index, which can exceed the default system limit.

**Quick fix** - increase the limit for your current terminal session:

```bash
ulimit -n 10240
```

**Permanent fix** - add to your `~/.zshrc` or `~/.bashrc`:

```bash
ulimit -n 10240
```

To check your current limits:

```bash
ulimit -n        # Current soft limit (often 256 on macOS)
ulimit -Hn       # Hard limit (max you can set)
```

### Buffer manager exception: Unable to allocate memory

If you encounter an error like:

```
Buffer manager exception: Unable to allocate memory! The buffer pool is full and no memory could be freed!
```

This is a Kuzu graph database buffer pool limit issue. The buffer pool may fill up during concurrent graph operations.

**Option 1: Clear Cognee data** (try first):

```bash
rm -rf .venv/lib/python3.12/site-packages/cognee/.cognee_system/
```

**Option 2: Increase Kuzu buffer pool size** (if Option 1 doesn't help):

Edit `.venv/lib/python3.12/site-packages/cognee/infrastructure/databases/graph/kuzu/adapter.py` and change all occurrences of buffer pool settings:

```python
# Change from:
buffer_pool_size=2048 * 1024 * 1024,  # 2GB
max_db_size=4096 * 1024 * 1024,       # 4GB

# To:
buffer_pool_size=8 * 1024 * 1024 * 1024,  # 8GB
max_db_size=16 * 1024 * 1024 * 1024,   # 16GB
```

Note: This change will be lost if you reinstall/update Cognee.
