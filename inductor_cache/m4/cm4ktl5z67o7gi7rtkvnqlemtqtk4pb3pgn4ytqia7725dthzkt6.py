
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr8': '*fp32', 'out_ptr10': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18), 'tt.equal_to': (17,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_clamp_div_floor_max_maximum_min_minimum_mul_sign_sub_0', 'mutated_arg_names': ['in_ptr7', 'in_ptr9', 'out_ptr10', 'out_ptr8'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 10, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_add_clamp_div_floor_max_maximum_min_minimum_mul_sign_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr4, out_ptr5, out_ptr6, out_ptr8, out_ptr10, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp6 = tl.load(in_ptr1 + (r0), None)
    tmp12 = tl.load(in_ptr2 + (0))
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.load(in_ptr3 + (0))
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp30 = tl.load(in_ptr4 + (0))
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.load(in_ptr5 + (0))
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp55 = tl.load(in_ptr6 + (0))
    tmp56 = tl.broadcast_to(tmp55, [1])
    tmp62 = tl.load(in_ptr7 + (0))
    tmp63 = tl.broadcast_to(tmp62, [1])
    tmp68 = tl.load(in_ptr8 + (0))
    tmp69 = tl.broadcast_to(tmp68, [1])
    tmp73 = tl.load(in_ptr9 + (0))
    tmp74 = tl.broadcast_to(tmp73, [1])
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = triton_helpers.promote_to_tensor(triton_helpers.min2(tmp1, 0))
    tmp5 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp1, 0))
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(triton_helpers.min2(tmp7, 0))
    tmp11 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp7, 0))
    tmp14 = tmp0 / tmp13
    tmp17 = tmp14 - tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = tmp18 < tmp17
    tmp20 = tmp19.to(tl.int8)
    tmp21 = tmp17 < tmp18
    tmp22 = tmp21.to(tl.int8)
    tmp23 = tmp20 - tmp22
    tmp24 = tmp23.to(tmp17.dtype)
    tmp25 = tl_math.abs(tmp17)
    tmp26 = 0.5
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.floor(tmp27)
    tmp29 = tmp24 * tmp28
    tmp32 = triton_helpers.maximum(tmp29, tmp31)
    tmp35 = triton_helpers.minimum(tmp32, tmp34)
    tmp36 = tmp35 + tmp16
    tmp37 = tmp36 * tmp13
    tmp38 = tmp6 / tmp13
    tmp39 = tmp38 - tmp16
    tmp40 = tmp18 < tmp39
    tmp41 = tmp40.to(tl.int8)
    tmp42 = tmp39 < tmp18
    tmp43 = tmp42.to(tl.int8)
    tmp44 = tmp41 - tmp43
    tmp45 = tmp44.to(tmp39.dtype)
    tmp46 = tl_math.abs(tmp39)
    tmp47 = tmp46 + tmp26
    tmp48 = libdevice.floor(tmp47)
    tmp49 = tmp45 * tmp48
    tmp50 = triton_helpers.maximum(tmp49, tmp31)
    tmp51 = triton_helpers.minimum(tmp50, tmp34)
    tmp52 = tmp51 + tmp16
    tmp53 = tmp52 * tmp13
    tmp54 = tmp37 + tmp53
    tmp57 = 0.9
    tmp58 = tmp56 * tmp57
    tmp59 = 0.1
    tmp60 = tmp3 * tmp59
    tmp61 = tmp58 + tmp60
    tmp64 = tmp63 * tmp57
    tmp65 = tmp9 * tmp59
    tmp66 = tmp64 + tmp65
    tmp67 = triton_helpers.minimum(tmp61, tmp66)
    tmp70 = tmp69 * tmp57
    tmp71 = tmp5 * tmp59
    tmp72 = tmp70 + tmp71
    tmp75 = tmp74 * tmp57
    tmp76 = tmp11 * tmp59
    tmp77 = tmp75 + tmp76
    tmp78 = triton_helpers.maximum(tmp72, tmp77)
    tl.store(out_ptr4 + (tl.broadcast_to(r0, [RBLOCK])), tmp54, None)
    tl.store(out_ptr5 + (tl.full([1], 0, tl.int32)), tmp67, None)
    tl.store(out_ptr6 + (tl.full([1], 0, tl.int32)), tmp78, None)
    tl.store(out_ptr8 + (tl.full([1], 0, tl.int32)), tmp66, None)
    tl.store(out_ptr10 + (tl.full([1], 0, tl.int32)), tmp77, None)
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp3, None)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp5, None)
