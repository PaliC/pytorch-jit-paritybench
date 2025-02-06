
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_avg_pool2d_mean_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_avg_pool2d_mean_sub_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 2)
    r1 = rindex // 2
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4 + 2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (5 + 2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (1 + 2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (4 + 2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (5 + 2*r0 + 8*r1), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp11 = tmp10 + tmp9
    tmp13 = tmp12 + tmp11
    tmp15 = tmp14 + tmp13
    tmp16 = tmp15 * tmp7
    tmp17 = tmp8 - tmp16
    tmp18 = tl_math.abs(tmp17)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.sum(tmp19, 1)[:, None]
    tl.store(out_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp8, None)
    tl.store(out_ptr1 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp16, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp21, None)
