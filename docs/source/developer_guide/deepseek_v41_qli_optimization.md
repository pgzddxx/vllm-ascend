# DeepSeek V4.1 QLI optimization on A3

This integration ports the QLI/QSLI implementation from
[upstream PR #21394](https://github.com/vllm-project/vllm-ascend/pull/21394),
frozen at `6c74291e4d04dfc64836cfb904a90fd9f731c2ff`, onto this fork's
`codex/operator-integration` branch. It updates the native operator, rather
than substituting a Python reference implementation.

## Scope and interface

The upstream changes update A3 cube/vector scheduling, candidate processing,
and QLI/QSLI metadata. This port includes the corresponding operator headers,
ACLNN APIs, AICPU metadata implementation, Torch adapter and dispatcher schema.
The existing A3 TND/PA_BBND contract and strided index-K caches are retained.

`npu_quant_lightning_indexer_v3` now accepts optional
`candidate_block_length` and returns four tensors:

```text
position_indices, position_values, candidate_indices, candidate_block_lengths
```

For a TND source call (`candidate_mode=1`), the last two tensors are INT32
with shapes `[T, Nk, candidate_topk_blocks]` and `[T, Nk]`. Modes 2 and 3
return empty candidate output tensors. Existing callers unpacking three
outputs must be updated and the native extension must be rebuilt together
with the custom operator package.

The model accepts the four-output ABI and uses the updated QLI kernel.
It deliberately does not pass the length output into A3 consumer calls.
The separate A3 `candidate_block_length`/native QSLI model wiring and the
experimental local C3 optimization are excluded from this integration.
The native length interface remains available for separate validation.

Two A3 build/runtime adaptations accompany the upstream port:

- The AICPU link target depends on both custom AICPU targets and all object
  files, so rebuilding does not reuse an outdated metadata library.
- The causal-length numerator in `GetS2BaseBlockNumOnMask` is materialized
  as a `volatile int32_t` before division. On the tested CANN 9.2.1 A3
  build, the original expression hung for a real prefill shape; the
  equivalent scalar expression passed that reproducer and service runs.
  The arithmetic formula is unchanged. A compiler root cause has not been
  established.

## Build and verify

Use a fresh checkout in an A3 main container with paired CANN and torch_npu
versions. The previous hardware validation used
`nightly-ci-main-a3-openeuler-cann9.2.1-20261111-r92` on openEuler/aarch64,
with CANN 9.2.1, torch 2.10.0 and torch_npu 2.10.0.post7.
Use the same vLLM version and service configuration as the baseline.

From the repository root inside that container:

```bash
export COMPILE_CUSTOM_KERNELS=1
python -m pip install -v -e . --no-deps --no-build-isolation

python -m pytest -v \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_qli_meta_registration.py

python -m pytest -v \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_quant_lightning_indexer_v2.py \
  -k 'candidate_vector_repeats or adaptive_topk_boundaries or consumer_preserves_each_query_mask or prefill_causal_loop_bounds'
```

The registration test loads the actual compiled extension and checks Meta
and FakeTensor dispatch, including source-length shapes. The NPU cases
compare selected scores with a reference, validate causal visibility,
candidate padding and per-query consumer masks, and check the source's
reported valid block count. They cover 8K/32K contexts, compression ratios
1/2 and candidate modes 1/2/3.
The prefill regression exercises Q `[2048, 32, 128]`, two 1024-query
sequences, compressed K lengths `[2048, 2560]`, ratio 2 and mode 3 with
the strided PA cache. Its completion and causal bounds protect the scalar
workaround. The newly added regression wrapper has not yet been run on NPU;
the archived reproducer used the same geometry.

Four outputs from the loaded dispatcher distinguish this runtime from the
old three-output ABI. To establish model execution, also verify QLI device
events in a short profile and the loaded library paths. An ABI check alone
does not prove that a model request executed the kernel. Since A3 consumer
length wiring is excluded, model profiles should not be interpreted as
evidence of a nonempty length-input/native-QSLI consumer path.

## Existing A3 evidence and limits

The 45 changed runtime/build files in this port were checked against the
final tested minimal-port source archive and match its Git blobs exactly.
Documentation and ABI test updates were added after that run. This is
reused hardware evidence, not a new full build or model test of the target
branch with its newer HC/Engram changes.

The archived minimal-port run passed 102 operator correctness cases. Eight
input snapshots covered real prefill, 8K/32K mode 1/3 and decode. Selected
index sets and candidate sets matched the baseline; real prefill and 8K
were bit-exact. Some near-tied 32K/decode indices changed order while their
sets remained equal. This does not replace dataset accuracy validation.
It also does not establish that earlier standalone 8K latency regressions
have been resolved.

Short profiles of matched complete model steps contained eight QLI calls:

| Role | Baseline QLI sum | Minimal port QLI sum | Reduction |
| --- | ---: | ---: | ---: |
| Prefill | 7.6804 ms | 6.9937 ms | 0.6866 ms |
| Decode | 3.0150 ms | 1.6214 ms | 1.3937 ms |

These are summed QLI device durations within each role, not end-to-end
latency reductions. Prefill and decode values must not be added together.
Single-round pure-P, pure-D and 1P1D throughput changes were +0.08%, +0.43%
and +0.29%; no stable end-to-end improvement or 4 ms saving was established.
GPQA was not run.

The source archive is identified by SHA256
`cb9a04176f7f82516cc37c8d79d79765ce4fec25b73aa8c1187f4bfb82cc291d`.
Its final minimal-port patch has SHA256
`d977bb5b32b51b371cf92bf006c81a90baa95b5958fda6cf9614fb558808f3f4`.
Keep the baseline/candidate service configuration identical when repeating
performance and accuracy validation on the integrated branch.
