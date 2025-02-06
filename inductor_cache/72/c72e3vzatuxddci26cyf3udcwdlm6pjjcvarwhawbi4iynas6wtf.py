
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_clamp_div_erf_log2_mul_neg_reciprocal_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_clamp_div_erf_log2_mul_neg_reciprocal_sub_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask)
    tmp28 = tl.load(in_ptr4 + (x0), xmask)
    tmp29 = tl.load(in_ptr5 + (x0), xmask)
    tmp31 = tl.load(in_ptr6 + (x0), xmask)
    tmp49 = tl.load(in_ptr7 + (x0), xmask)
    tmp50 = tl.load(in_ptr8 + (x0), xmask)
    tmp52 = tl.load(in_ptr9 + (x0), xmask)
    tmp70 = tl.load(in_ptr10 + (x0), xmask)
    tmp71 = tl.load(in_ptr11 + (x0), xmask)
    tmp73 = tl.load(in_ptr12 + (x0), xmask)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 - tmp4
    tmp7 = 1e-09
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp5 * tmp10
    tmp12 = 0.7071067811865475
    tmp13 = tmp11 * tmp12
    tmp14 = libdevice.erf(tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16 * tmp2
    tmp18 = tmp1 - tmp2
    tmp19 = tmp18 - tmp4
    tmp20 = tmp19 * tmp10
    tmp21 = tmp20 * tmp12
    tmp22 = libdevice.erf(tmp21)
    tmp23 = tmp22 + tmp15
    tmp24 = tmp23 * tmp2
    tmp25 = tmp17 - tmp24
    tmp26 = tl_math.abs(tmp25)
    tmp27 = tmp0 * tmp26
    tmp30 = tmp3 - tmp29
    tmp32 = triton_helpers.maximum(tmp31, tmp7)
    tmp33 = tmp9 / tmp32
    tmp34 = tmp30 * tmp33
    tmp35 = tmp34 * tmp12
    tmp36 = libdevice.erf(tmp35)
    tmp37 = tmp36 + tmp15
    tmp38 = tmp37 * tmp2
    tmp39 = tmp18 - tmp29
    tmp40 = tmp39 * tmp33
    tmp41 = tmp40 * tmp12
    tmp42 = libdevice.erf(tmp41)
    tmp43 = tmp42 + tmp15
    tmp44 = tmp43 * tmp2
    tmp45 = tmp38 - tmp44
    tmp46 = tl_math.abs(tmp45)
    tmp47 = tmp28 * tmp46
    tmp48 = tmp27 + tmp47
    tmp51 = tmp3 - tmp50
    tmp53 = triton_helpers.maximum(tmp52, tmp7)
    tmp54 = tmp9 / tmp53
    tmp55 = tmp51 * tmp54
    tmp56 = tmp55 * tmp12
    tmp57 = libdevice.erf(tmp56)
    tmp58 = tmp57 + tmp15
    tmp59 = tmp58 * tmp2
    tmp60 = tmp18 - tmp50
    tmp61 = tmp60 * tmp54
    tmp62 = tmp61 * tmp12
    tmp63 = libdevice.erf(tmp62)
    tmp64 = tmp63 + tmp15
    tmp65 = tmp64 * tmp2
    tmp66 = tmp59 - tmp65
    tmp67 = tl_math.abs(tmp66)
    tmp68 = tmp49 * tmp67
    tmp69 = tmp48 + tmp68
    tmp72 = tmp3 - tmp71
    tmp74 = triton_helpers.maximum(tmp73, tmp7)
    tmp75 = tmp9 / tmp74
    tmp76 = tmp72 * tmp75
    tmp77 = tmp76 * tmp12
    tmp78 = libdevice.erf(tmp77)
    tmp79 = tmp78 + tmp15
    tmp80 = tmp79 * tmp2
    tmp81 = tmp18 - tmp71
    tmp82 = tmp81 * tmp75
    tmp83 = tmp82 * tmp12
    tmp84 = libdevice.erf(tmp83)
    tmp85 = tmp84 + tmp15
    tmp86 = tmp85 * tmp2
    tmp87 = tmp80 - tmp86
    tmp88 = tl_math.abs(tmp87)
    tmp89 = tmp70 * tmp88
    tmp90 = tmp69 + tmp89
    tmp91 = 1e-06
    tmp92 = triton_helpers.maximum(tmp90, tmp91)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tmp94 = libdevice.log2(tmp93)
    tmp95 = -tmp94
    tl.store(in_out_ptr0 + (x0), tmp95, xmask)
