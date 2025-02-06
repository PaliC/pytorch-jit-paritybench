
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_mul_sigmoid_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_mul_sigmoid_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full([1], -2, tl.int64)
    tmp19 = tl.full([1], 0, tl.int64)
    tmp20 = tmp18 >= tmp19
    tmp21 = tl.full([1], 1, tl.int64)
    tmp22 = tmp18 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tmp23 & tmp23
    tmp25 = tl.full([1], -1, tl.int64)
    tmp26 = tmp25 >= tmp19
    tmp27 = tmp25 < tmp21
    tmp28 = tmp26 & tmp27
    tmp29 = tmp23 & tmp28
    tmp30 = tmp17 > tmp17
    tmp31 = tl.full([1], 1, tl.int8)
    tmp32 = tl.full([1], 0, tl.int8)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = triton_helpers.maximum(tmp17, tmp17)
    tmp35 = tmp19 >= tmp19
    tmp36 = tmp19 < tmp21
    tmp37 = tmp35 & tmp36
    tmp38 = tmp23 & tmp37
    tmp39 = tmp17 > tmp34
    tmp40 = tl.full([1], 2, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp33)
    tmp42 = triton_helpers.maximum(tmp17, tmp34)
    tmp43 = tmp21 >= tmp19
    tmp44 = tmp21 < tmp21
    tmp45 = tmp43 & tmp44
    tmp46 = tmp23 & tmp45
    tmp47 = tmp17 > tmp42
    tmp48 = tl.full([1], 3, tl.int8)
    tmp49 = tl.where(tmp47, tmp48, tmp41)
    tmp50 = triton_helpers.maximum(tmp17, tmp42)
    tmp51 = tl.full([1], 2, tl.int64)
    tmp52 = tmp51 >= tmp19
    tmp53 = tmp51 < tmp21
    tmp54 = tmp52 & tmp53
    tmp55 = tmp23 & tmp54
    tmp56 = tmp17 > tmp50
    tmp57 = tl.full([1], 4, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp49)
    tmp59 = triton_helpers.maximum(tmp17, tmp50)
    tmp60 = tmp28 & tmp23
    tmp61 = tmp17 > tmp59
    tmp62 = tl.full([1], 5, tl.int8)
    tmp63 = tl.where(tmp61, tmp62, tmp58)
    tmp64 = triton_helpers.maximum(tmp17, tmp59)
    tmp65 = tmp28 & tmp28
    tmp66 = tmp17 > tmp64
    tmp67 = tl.full([1], 6, tl.int8)
    tmp68 = tl.where(tmp66, tmp67, tmp63)
    tmp69 = triton_helpers.maximum(tmp17, tmp64)
    tmp70 = tmp28 & tmp37
    tmp71 = tmp17 > tmp69
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp68)
    tmp74 = triton_helpers.maximum(tmp17, tmp69)
    tmp75 = tmp28 & tmp45
    tmp76 = tmp17 > tmp74
    tmp77 = tl.full([1], 8, tl.int8)
    tmp78 = tl.where(tmp76, tmp77, tmp73)
    tmp79 = triton_helpers.maximum(tmp17, tmp74)
    tmp80 = tmp28 & tmp54
    tmp81 = tmp17 > tmp79
    tmp82 = tl.full([1], 9, tl.int8)
    tmp83 = tl.where(tmp81, tmp82, tmp78)
    tmp84 = triton_helpers.maximum(tmp17, tmp79)
    tmp85 = tmp37 & tmp23
    tmp86 = tmp17 > tmp84
    tmp87 = tl.full([1], 10, tl.int8)
    tmp88 = tl.where(tmp86, tmp87, tmp83)
    tmp89 = triton_helpers.maximum(tmp17, tmp84)
    tmp90 = tmp37 & tmp28
    tmp91 = tmp17 > tmp89
    tmp92 = tl.full([1], 11, tl.int8)
    tmp93 = tl.where(tmp91, tmp92, tmp88)
    tmp94 = triton_helpers.maximum(tmp17, tmp89)
    tmp95 = tmp37 & tmp37
    tmp96 = tmp17 > tmp94
    tmp97 = tl.full([1], 12, tl.int8)
    tmp98 = tl.where(tmp96, tmp97, tmp93)
    tmp99 = triton_helpers.maximum(tmp17, tmp94)
    tmp100 = tmp37 & tmp45
    tmp101 = tmp17 > tmp99
    tmp102 = tl.full([1], 13, tl.int8)
    tmp103 = tl.where(tmp101, tmp102, tmp98)
    tmp104 = triton_helpers.maximum(tmp17, tmp99)
    tmp105 = tmp37 & tmp54
    tmp106 = tmp17 > tmp104
    tmp107 = tl.full([1], 14, tl.int8)
    tmp108 = tl.where(tmp106, tmp107, tmp103)
    tmp109 = triton_helpers.maximum(tmp17, tmp104)
    tmp110 = tmp45 & tmp23
    tmp111 = tmp17 > tmp109
    tmp112 = tl.full([1], 15, tl.int8)
    tmp113 = tl.where(tmp111, tmp112, tmp108)
    tmp114 = triton_helpers.maximum(tmp17, tmp109)
    tmp115 = tmp45 & tmp28
    tmp116 = tmp17 > tmp114
    tmp117 = tl.full([1], 16, tl.int8)
    tmp118 = tl.where(tmp116, tmp117, tmp113)
    tmp119 = triton_helpers.maximum(tmp17, tmp114)
    tmp120 = tmp45 & tmp37
    tmp121 = tmp17 > tmp119
    tmp122 = tl.full([1], 17, tl.int8)
    tmp123 = tl.where(tmp121, tmp122, tmp118)
    tmp124 = triton_helpers.maximum(tmp17, tmp119)
    tmp125 = tmp45 & tmp45
    tmp126 = tmp17 > tmp124
    tmp127 = tl.full([1], 18, tl.int8)
    tmp128 = tl.where(tmp126, tmp127, tmp123)
    tmp129 = triton_helpers.maximum(tmp17, tmp124)
    tmp130 = tmp45 & tmp54
    tmp131 = tmp17 > tmp129
    tmp132 = tl.full([1], 19, tl.int8)
    tmp133 = tl.where(tmp131, tmp132, tmp128)
    tmp134 = triton_helpers.maximum(tmp17, tmp129)
    tmp135 = tmp54 & tmp23
    tmp136 = tmp17 > tmp134
    tmp137 = tl.full([1], 20, tl.int8)
    tmp138 = tl.where(tmp136, tmp137, tmp133)
    tmp139 = triton_helpers.maximum(tmp17, tmp134)
    tmp140 = tmp54 & tmp28
    tmp141 = tmp17 > tmp139
    tmp142 = tl.full([1], 21, tl.int8)
    tmp143 = tl.where(tmp141, tmp142, tmp138)
    tmp144 = triton_helpers.maximum(tmp17, tmp139)
    tmp145 = tmp54 & tmp37
    tmp146 = tmp17 > tmp144
    tmp147 = tl.full([1], 22, tl.int8)
    tmp148 = tl.where(tmp146, tmp147, tmp143)
    tmp149 = triton_helpers.maximum(tmp17, tmp144)
    tmp150 = tmp54 & tmp45
    tmp151 = tmp17 > tmp149
    tmp152 = tl.full([1], 23, tl.int8)
    tmp153 = tl.where(tmp151, tmp152, tmp148)
    tmp154 = triton_helpers.maximum(tmp17, tmp149)
    tmp155 = tmp54 & tmp54
    tmp156 = tmp17 > tmp154
    tmp157 = tl.full([1], 24, tl.int8)
    tmp158 = tl.where(tmp156, tmp157, tmp153)
    tmp159 = triton_helpers.maximum(tmp17, tmp154)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
    tl.store(out_ptr0 + (x2), tmp158, xmask)
