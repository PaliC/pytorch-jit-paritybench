
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x2 = xindex // 64
    x3 = xindex
    x4 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (3 + 6*x0 + 96*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (4 + 6*x0 + 96*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (5 + 6*x0 + 96*x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (6*x0 + 96*x2), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (1 + 6*x0 + 96*x2), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (2 + 6*x0 + 96*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 2.0
    tmp6 = tmp4 * tmp5
    tmp7 = 1.5
    tmp8 = tmp6 + tmp7
    tmp9 = -0.5
    tmp10 = tmp8 - tmp9
    tmp11 = tl_math.abs(tmp10)
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = libdevice.floor(tmp13)
    tmp15 = tmp14.to(tl.int8)
    tmp16 = tl.full([1], 1, tl.int8)
    tmp17 = tmp15 & tmp16
    tmp18 = tl.full([1], 0, tl.int8)
    tmp19 = tmp17 == tmp18
    tmp20 = 4.0
    tmp21 = libdevice.fmod(tmp11, tmp20)
    tmp22 = tmp21 + tmp9
    tmp23 = 3.5
    tmp24 = tmp23 - tmp21
    tmp25 = tl.where(tmp19, tmp22, tmp24)
    tmp26 = 0.0
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp28 = 3.0
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = libdevice.floor(tmp29)
    tmp33 = tmp31 + tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp35 * tmp5
    tmp37 = tmp36 + tmp7
    tmp38 = tmp37 - tmp9
    tmp39 = tl_math.abs(tmp38)
    tmp40 = tmp39 * tmp12
    tmp41 = libdevice.floor(tmp40)
    tmp42 = tmp41.to(tl.int8)
    tmp43 = tmp42 & tmp16
    tmp44 = tmp43 == tmp18
    tmp45 = libdevice.fmod(tmp39, tmp20)
    tmp46 = tmp45 + tmp9
    tmp47 = tmp23 - tmp45
    tmp48 = tl.where(tmp44, tmp46, tmp47)
    tmp49 = triton_helpers.maximum(tmp48, tmp26)
    tmp50 = triton_helpers.minimum(tmp49, tmp28)
    tmp51 = libdevice.floor(tmp50)
    tmp52 = 1.0
    tmp53 = tmp51 + tmp52
    tmp54 = tmp53 - tmp50
    tmp55 = tmp29 - tmp30
    tmp56 = tmp30 + tmp52
    tmp57 = tmp56 - tmp29
    tmp58 = tmp50 - tmp51
    tmp59 = tmp51 >= tmp26
    tmp60 = tmp51 < tmp20
    tmp61 = tmp30 >= tmp26
    tmp62 = tmp30 < tmp20
    tmp63 = tmp61 & tmp62
    tmp64 = tmp60 & tmp63
    tmp65 = tmp59 & tmp64
    tmp66 = tmp30.to(tl.int64)
    tmp67 = tl.full([1], 0, tl.int64)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tl.full([XBLOCK], 4, tl.int32)
    tmp70 = tmp68 + tmp69
    tmp71 = tmp68 < 0
    tmp72 = tl.where(tmp71, tmp70, tmp68)
    tl.device_assert(((0 <= tmp72) & (tmp72 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp72 < 4")
    tmp74 = tmp51.to(tl.int64)
    tmp75 = tl.where(tmp65, tmp74, tmp67)
    tmp76 = tmp75 + tmp69
    tmp77 = tmp75 < 0
    tmp78 = tl.where(tmp77, tmp76, tmp75)
    tl.device_assert(((0 <= tmp78) & (tmp78 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp78 < 4")
    tmp80 = tl.load(in_ptr1 + (tmp78 + 4*tmp72 + 16*x4), xmask, eviction_policy='evict_last')
    tmp81 = tmp54 * tmp57
    tmp82 = tl.where(tmp65, tmp81, tmp26)
    tmp83 = tmp80 * tmp82
    tmp84 = tmp53 >= tmp26
    tmp85 = tmp53 < tmp20
    tmp86 = tmp85 & tmp63
    tmp87 = tmp84 & tmp86
    tmp88 = tl.where(tmp87, tmp66, tmp67)
    tmp89 = tmp88 + tmp69
    tmp90 = tmp88 < 0
    tmp91 = tl.where(tmp90, tmp89, tmp88)
    tl.device_assert(((0 <= tmp91) & (tmp91 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp91 < 4")
    tmp93 = tmp53.to(tl.int64)
    tmp94 = tl.where(tmp87, tmp93, tmp67)
    tmp95 = tmp94 + tmp69
    tmp96 = tmp94 < 0
    tmp97 = tl.where(tmp96, tmp95, tmp94)
    tl.device_assert(((0 <= tmp97) & (tmp97 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp97 < 4")
    tmp99 = tl.load(in_ptr1 + (tmp97 + 4*tmp91 + 16*x4), xmask, eviction_policy='evict_last')
    tmp100 = tmp58 * tmp57
    tmp101 = tl.where(tmp87, tmp100, tmp26)
    tmp102 = tmp99 * tmp101
    tmp103 = tmp83 + tmp102
    tmp104 = tmp56 >= tmp26
    tmp105 = tmp56 < tmp20
    tmp106 = tmp104 & tmp105
    tmp107 = tmp60 & tmp106
    tmp108 = tmp59 & tmp107
    tmp109 = tmp56.to(tl.int64)
    tmp110 = tl.where(tmp108, tmp109, tmp67)
    tmp111 = tmp110 + tmp69
    tmp112 = tmp110 < 0
    tmp113 = tl.where(tmp112, tmp111, tmp110)
    tl.device_assert(((0 <= tmp113) & (tmp113 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp113 < 4")
    tmp115 = tl.where(tmp108, tmp74, tmp67)
    tmp116 = tmp115 + tmp69
    tmp117 = tmp115 < 0
    tmp118 = tl.where(tmp117, tmp116, tmp115)
    tl.device_assert(((0 <= tmp118) & (tmp118 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp118 < 4")
    tmp120 = tl.load(in_ptr1 + (tmp118 + 4*tmp113 + 16*x4), xmask, eviction_policy='evict_last')
    tmp121 = tmp54 * tmp55
    tmp122 = tl.where(tmp108, tmp121, tmp26)
    tmp123 = tmp120 * tmp122
    tmp124 = tmp103 + tmp123
    tmp125 = tmp85 & tmp106
    tmp126 = tmp84 & tmp125
    tmp127 = tl.where(tmp126, tmp109, tmp67)
    tmp128 = tmp127 + tmp69
    tmp129 = tmp127 < 0
    tmp130 = tl.where(tmp129, tmp128, tmp127)
    tl.device_assert(((0 <= tmp130) & (tmp130 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp130 < 4")
    tmp132 = tl.where(tmp126, tmp93, tmp67)
    tmp133 = tmp132 + tmp69
    tmp134 = tmp132 < 0
    tmp135 = tl.where(tmp134, tmp133, tmp132)
    tl.device_assert(((0 <= tmp135) & (tmp135 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp135 < 4")
    tmp137 = tl.load(in_ptr1 + (tmp135 + 4*tmp130 + 16*x4), xmask, eviction_policy='evict_last')
    tmp138 = tmp58 * tmp55
    tmp139 = tl.where(tmp126, tmp138, tmp26)
    tmp140 = tmp137 * tmp139
    tmp141 = tmp124 + tmp140
    tl.store(in_out_ptr0 + (x3), tmp141, xmask)
