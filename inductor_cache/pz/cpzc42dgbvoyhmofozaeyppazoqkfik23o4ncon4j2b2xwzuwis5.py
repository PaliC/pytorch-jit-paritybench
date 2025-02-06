
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mean_mul_pow_sub_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_mean_mul_pow_sub_5(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp36 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        r1 = rindex // 4
        tmp0 = tl.load(in_ptr0 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (4*r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (1 + 4*r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr0 + (2 + 4*r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr0 + (3 + 4*r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr1 + (4*r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr1 + (1 + 4*r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr1 + (2 + 4*r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.load(in_ptr1 + (3 + 4*r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1 * tmp1
        tmp4 = tmp3 * tmp3
        tmp5 = tmp2 + tmp4
        tmp7 = tmp6 * tmp6
        tmp8 = tmp5 + tmp7
        tmp10 = tmp9 * tmp9
        tmp11 = tmp8 + tmp10
        tmp12 = libdevice.sqrt(tmp11)
        tmp13 = 1e-12
        tmp14 = triton_helpers.maximum(tmp12, tmp13)
        tmp15 = tmp0 / tmp14
        tmp17 = tmp15 - tmp16
        tmp18 = tmp16 + tmp17
        tmp20 = tmp19 * tmp19
        tmp22 = tmp21 * tmp21
        tmp23 = tmp20 + tmp22
        tmp25 = tmp24 * tmp24
        tmp26 = tmp23 + tmp25
        tmp28 = tmp27 * tmp27
        tmp29 = tmp26 + tmp28
        tmp30 = libdevice.sqrt(tmp29)
        tmp31 = triton_helpers.maximum(tmp30, tmp13)
        tmp32 = tmp16 / tmp31
        tmp33 = tmp15 - tmp32
        tmp34 = tmp33 * tmp33
        tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
        tmp37 = _tmp36 + tmp35
        _tmp36 = tl.where(rmask, tmp37, _tmp36)
        tl.store(out_ptr1 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp18, rmask)
    tmp36 = tl.sum(_tmp36, 1)[:, None]
    tmp38 = 256.0
    tmp39 = tmp36 / tmp38
    tmp40 = 0.25
    tmp41 = tmp39 * tmp40
    tmp42 = tmp41 + tmp39
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp42, None)
