
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*i64', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_exp_min_neg_sub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_div_exp_min_neg_sub_3(in_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    _tmp5_index = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = libdevice.sqrt(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = triton_helpers.minimum(_tmp3, tmp2)
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
        _tmp5_next, _tmp5_index_next = triton_helpers.minimum_with_index(
            _tmp5, _tmp5_index, tmp2, rindex
        )
        _tmp5 = tl.where(rmask & xmask, _tmp5_next, _tmp5)
        _tmp5_index = tl.where(rmask & xmask, _tmp5_index_next, _tmp5_index)
    tmp3 = triton_helpers.min2(_tmp3, 1)[:, None]
    tmp5_val, tmp5_idx = triton_helpers.min_with_index(_tmp5, _tmp5_index, 1)
    tmp5 = tmp5_idx[:, None]
    tl.store(out_ptr1 + (x0), tmp5, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = libdevice.sqrt(tmp6)
        tmp8 = tmp7 - tmp3
        tmp9 = -tmp8
        tmp10 = 0.2
        tmp11 = tmp9 * tmp10
        tmp12 = tl_math.exp(tmp11)
        tl.store(out_ptr2 + (r1 + 4096*x0), tmp12, rmask & xmask)
