
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10), 'tt.equal_to': (9,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_eq_mul_smooth_l1_loss_sum_11', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_div_eq_mul_smooth_l1_loss_sum_11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp67 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp70 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r6 = rindex // 16
        r1 = ((rindex // 4) % 4)
        r3 = rindex // 64
        r0 = (rindex % 4)
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + (r6), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr3 + (r4), rmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_out_ptr0 + (r4), rmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tl.load(in_ptr4 + (r4), rmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tl.load(in_ptr0 + (r4 // 16), rmask, eviction_policy='evict_last', other=0.0)
        tmp53 = tl.load(in_ptr5 + (r4), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 4, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp4 < 4")
        tmp6 = tl.load(in_ptr1 + (r1 + 4*tmp4 + 16*r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp6 + tmp1
        tmp8 = tmp6 < 0
        tmp9 = tl.where(tmp8, tmp7, tmp6)
        tl.device_assert(((0 <= tmp9) & (tmp9 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp9 < 4")
        tmp11 = tl.load(in_ptr2 + (tmp4 + 4*r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (tmp9 + 4*r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.load(in_ptr1 + (r0 + 4*tmp4 + 16*r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp14 + tmp1
        tmp16 = tmp14 < 0
        tmp17 = tl.where(tmp16, tmp15, tmp14)
        tl.device_assert(((0 <= tmp17) & (tmp17 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp17 < 4")
        tmp19 = tl.load(in_ptr2 + (tmp17 + 4*r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp11 + tmp19
        tmp21 = tmp13 * tmp20
        tmp22 = 4.0
        tmp23 = tmp21 == tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp24 * tmp25
        tmp28 = tmp27 * tmp26
        tmp31 = tmp30 + tmp1
        tmp32 = tmp30 < 0
        tmp33 = tl.where(tmp32, tmp31, tmp30)
        tl.device_assert(((0 <= tmp33) & (tmp33 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp33 < 4")
        tmp35 = tl.load(in_ptr1 + (4*tmp33 + 16*(r4 // 64) + (((r4 // 4) % 4))), rmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tmp35 + tmp1
        tmp37 = tmp35 < 0
        tmp38 = tl.where(tmp37, tmp36, tmp35)
        tl.device_assert(((0 <= tmp38) & (tmp38 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp38 < 4")
        tmp40 = tl.load(in_ptr2 + (tmp33 + 4*(r4 // 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.load(in_ptr2 + (tmp38 + 4*(r4 // 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp42 = tmp40 + tmp41
        tmp43 = tl.load(in_ptr1 + (4*tmp33 + 16*(r4 // 64) + ((r4 % 4))), rmask, eviction_policy='evict_first', other=0.0)
        tmp44 = tmp43 + tmp1
        tmp45 = tmp43 < 0
        tmp46 = tl.where(tmp45, tmp44, tmp43)
        tl.device_assert(((0 <= tmp46) & (tmp46 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp46 < 4")
        tmp48 = tl.load(in_ptr2 + (tmp46 + 4*(r4 // 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp49 = tmp40 + tmp48
        tmp50 = tmp42 * tmp49
        tmp51 = tmp50 == tmp22
        tmp52 = tmp51.to(tl.float32)
        tmp54 = tmp52 * tmp53
        tmp55 = tmp29 * tmp54
        tmp56 = tmp28 - tmp55
        tmp57 = tl_math.abs(tmp56)
        tmp58 = 1.0
        tmp59 = tmp57 < tmp58
        tmp60 = tmp57 * tmp57
        tmp61 = 0.5
        tmp62 = tmp60 * tmp61
        tmp63 = tmp62 * tmp58
        tmp64 = tmp57 - tmp61
        tmp65 = tl.where(tmp59, tmp63, tmp64)
        tmp66 = tl.broadcast_to(tmp65, [XBLOCK, RBLOCK])
        tmp68 = _tmp67 + tmp66
        _tmp67 = tl.where(rmask, tmp68, _tmp67)
        tmp69 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp71 = _tmp70 + tmp69
        _tmp70 = tl.where(rmask, tmp71, _tmp70)
    tmp67 = tl.sum(_tmp67, 1)[:, None]
    tmp70 = tl.sum(_tmp70, 1)[:, None]
    tmp72 = tl.load(in_out_ptr1 + (0))
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, 1])
    tmp74 = tl.load(in_ptr6 + (0))
    tmp75 = tl.broadcast_to(tmp74, [XBLOCK, 1])
    tmp76 = tmp73 / tmp75
    tmp77 = tmp67 / tmp70
    tmp78 = tmp76 + tmp77
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp78, None)
