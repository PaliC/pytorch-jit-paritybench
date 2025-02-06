
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = ((rindex // 4) % 16)
        r1 = (rindex % 4)
        r3 = ((rindex // 16) % 4)
        r4 = rindex // 64
        r6 = (rindex % 16)
        r0 = rindex
        tmp4 = tl.load(in_ptr1 + (r5), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr0 + (r6 + 16*r4), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr0 + (64 + r6 + 16*r4), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.load(in_ptr0 + (128 + r6 + 16*r4), rmask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr0 + (192 + r6 + 16*r4), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.full([XBLOCK, RBLOCK], 16, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 16)) | ~(rmask), "index out of bounds: 0 <= tmp8 < 16")
        tmp10 = tl.load(in_ptr2 + (16*r1 + 64*r3 + ((tmp8 % 16))), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 * tmp11
        tmp14 = tmp13 * tmp11
        tmp15 = tmp12 - tmp14
        tmp16 = tl_math.abs(tmp15)
        tmp17 = 0.0001
        tmp18 = tmp2 + tmp17
        tmp19 = tmp16 / tmp18
        tmp21 = tmp10 * tmp20
        tmp22 = tmp13 * tmp20
        tmp23 = tmp21 - tmp22
        tmp24 = tl_math.abs(tmp23)
        tmp25 = tmp24 / tmp18
        tmp26 = tmp19 + tmp25
        tmp28 = tmp10 * tmp27
        tmp29 = tmp13 * tmp27
        tmp30 = tmp28 - tmp29
        tmp31 = tl_math.abs(tmp30)
        tmp32 = tmp31 / tmp18
        tmp33 = tmp26 + tmp32
        tmp35 = tmp10 * tmp34
        tmp36 = tmp13 * tmp34
        tmp37 = tmp35 - tmp36
        tmp38 = tl_math.abs(tmp37)
        tmp39 = tmp38 / tmp18
        tmp40 = tmp33 + tmp39
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp40, rmask)
