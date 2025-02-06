
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mul_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = xindex // 4
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + 16*x0 + 64*x2), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (r3 + 16*x4), xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (r3 + 16*x0 + 64*x2), xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr3 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (r3 + 16*x4), xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = libdevice.sqrt(tmp1)
    tmp3 = 1e-12
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = tmp0 / tmp4
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = triton_helpers.maximum(tmp8, tmp3)
    tmp10 = tmp6 / tmp9
    tmp11 = tmp5 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp18 = libdevice.sqrt(tmp17)
    tmp19 = triton_helpers.maximum(tmp18, tmp3)
    tmp20 = tmp16 / tmp19
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = triton_helpers.maximum(tmp23, tmp3)
    tmp25 = tmp21 / tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
    tmp29 = tl.where(xmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None]
    tmp31 = tmp20 * tmp10
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.where(xmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp15, xmask)
    tl.store(out_ptr1 + (x5), tmp30, xmask)
    tl.store(out_ptr2 + (x5), tmp35, xmask)
