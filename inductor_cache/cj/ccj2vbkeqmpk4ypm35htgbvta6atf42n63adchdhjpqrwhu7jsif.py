
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_index_put_lift_fresh_scatter_add_2', 'mutated_arg_names': ['in_ptr2', 'out_ptr3', 'out_ptr4'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_index_put_lift_fresh_scatter_add_2(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = xindex // 64
    x2 = xindex
    tmp149 = tl.load(in_ptr2 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr0 + (16*tmp8 + (((x0) % 16))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr1 + (16*tmp8 + 32*x1 + (((x0) % 16))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.floor(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 3.0
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp6 < tmp6
    tmp20 = tl.where(tmp19, tmp6, tmp5)
    tmp21 = tl.load(in_ptr0 + (16*tmp20 + (((x0) % 16))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr1 + (16*tmp20 + 32*x1 + (((x0) % 16))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.floor(tmp23)
    tmp25 = tmp24 + tmp13
    tmp26 = triton_helpers.maximum(tmp25, tmp15)
    tmp27 = triton_helpers.minimum(tmp26, tmp17)
    tmp28 = 4.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp18 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp4, tmp30, tmp31)
    tmp33 = tmp0 >= tmp3
    tmp34 = tl.full([1], 32, tl.int64)
    tmp35 = tmp0 < tmp34
    tmp36 = tmp33 & tmp35
    tmp37 = tl.full([1], 0, tl.int64)
    tmp38 = tl.full([1], 1, tl.int64)
    tmp39 = tmp37 < tmp38
    tmp40 = tl.where(tmp39, tmp38, tmp37)
    tmp41 = tl.load(in_ptr0 + (16*tmp40 + ((((-16) + x0) % 16))), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr1 + (16*tmp40 + 32*x1 + ((((-16) + x0) % 16))), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.floor(tmp43)
    tmp45 = 1.0
    tmp46 = tmp44 + tmp45
    tmp47 = 0.0
    tmp48 = triton_helpers.maximum(tmp46, tmp47)
    tmp49 = 3.0
    tmp50 = triton_helpers.minimum(tmp48, tmp49)
    tmp51 = tmp38 < tmp38
    tmp52 = tl.where(tmp51, tmp38, tmp37)
    tmp53 = tl.load(in_ptr0 + (16*tmp52 + ((((-16) + x0) % 16))), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr1 + (16*tmp52 + 32*x1 + ((((-16) + x0) % 16))), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 + tmp54
    tmp56 = libdevice.floor(tmp55)
    tmp57 = triton_helpers.maximum(tmp56, tmp47)
    tmp58 = triton_helpers.minimum(tmp57, tmp49)
    tmp59 = 4.0
    tmp60 = tmp58 * tmp59
    tmp61 = tmp50 + tmp60
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp36, tmp61, tmp62)
    tmp64 = tmp0 >= tmp34
    tmp65 = tl.full([1], 48, tl.int64)
    tmp66 = tmp0 < tmp65
    tmp67 = tmp64 & tmp66
    tmp68 = tl.full([1], 0, tl.int64)
    tmp69 = tl.full([1], 1, tl.int64)
    tmp70 = tmp68 < tmp69
    tmp71 = tl.where(tmp70, tmp69, tmp68)
    tmp72 = tl.load(in_ptr0 + (16*tmp71 + ((((-32) + x0) % 16))), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.load(in_ptr1 + (16*tmp71 + 32*x1 + ((((-32) + x0) % 16))), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp74 = tmp72 + tmp73
    tmp75 = libdevice.floor(tmp74)
    tmp76 = 0.0
    tmp77 = triton_helpers.maximum(tmp75, tmp76)
    tmp78 = 3.0
    tmp79 = triton_helpers.minimum(tmp77, tmp78)
    tmp80 = tmp69 < tmp69
    tmp81 = tl.where(tmp80, tmp69, tmp68)
    tmp82 = tl.load(in_ptr0 + (16*tmp81 + ((((-32) + x0) % 16))), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tl.load(in_ptr1 + (16*tmp81 + 32*x1 + ((((-32) + x0) % 16))), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp84 = tmp82 + tmp83
    tmp85 = libdevice.floor(tmp84)
    tmp86 = 1.0
    tmp87 = tmp85 + tmp86
    tmp88 = triton_helpers.maximum(tmp87, tmp76)
    tmp89 = triton_helpers.minimum(tmp88, tmp78)
    tmp90 = 4.0
    tmp91 = tmp89 * tmp90
    tmp92 = tmp79 + tmp91
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp67, tmp92, tmp93)
    tmp95 = tmp0 >= tmp65
    tmp96 = tl.full([1], 64, tl.int64)
    tmp97 = tmp0 < tmp96
    tmp98 = tl.full([1], 0, tl.int64)
    tmp99 = tl.full([1], 1, tl.int64)
    tmp100 = tmp98 < tmp99
    tmp101 = tl.where(tmp100, tmp99, tmp98)
    tmp102 = tl.load(in_ptr0 + (16*tmp101 + ((((-48) + x0) % 16))), tmp95 & xmask, eviction_policy='evict_last', other=0.0)
    tmp103 = tl.load(in_ptr1 + (16*tmp101 + 32*x1 + ((((-48) + x0) % 16))), tmp95 & xmask, eviction_policy='evict_last', other=0.0)
    tmp104 = tmp102 + tmp103
    tmp105 = libdevice.floor(tmp104)
    tmp106 = 0.0
    tmp107 = triton_helpers.maximum(tmp105, tmp106)
    tmp108 = 3.0
    tmp109 = triton_helpers.minimum(tmp107, tmp108)
    tmp110 = tmp99 < tmp99
    tmp111 = tl.where(tmp110, tmp99, tmp98)
    tmp112 = tl.load(in_ptr0 + (16*tmp111 + ((((-48) + x0) % 16))), tmp95 & xmask, eviction_policy='evict_last', other=0.0)
    tmp113 = tl.load(in_ptr1 + (16*tmp111 + 32*x1 + ((((-48) + x0) % 16))), tmp95 & xmask, eviction_policy='evict_last', other=0.0)
    tmp114 = tmp112 + tmp113
    tmp115 = libdevice.floor(tmp114)
    tmp116 = triton_helpers.maximum(tmp115, tmp106)
    tmp117 = triton_helpers.minimum(tmp116, tmp108)
    tmp118 = 4.0
    tmp119 = tmp117 * tmp118
    tmp120 = tmp109 + tmp119
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp95, tmp120, tmp121)
    tmp123 = tl.where(tmp67, tmp94, tmp122)
    tmp124 = tl.where(tmp36, tmp63, tmp123)
    tmp125 = tl.where(tmp4, tmp32, tmp124)
    tmp126 = tmp14 != tmp18
    tmp127 = tmp25 != tmp27
    tmp128 = tmp126 | tmp127
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp4, tmp128, tmp129)
    tmp131 = tmp46 != tmp50
    tmp132 = tmp56 != tmp58
    tmp133 = tmp131 | tmp132
    tmp134 = tl.full(tmp133.shape, 0.0, tmp133.dtype)
    tmp135 = tl.where(tmp36, tmp133, tmp134)
    tmp136 = tmp75 != tmp79
    tmp137 = tmp87 != tmp89
    tmp138 = tmp136 | tmp137
    tmp139 = tl.full(tmp138.shape, 0.0, tmp138.dtype)
    tmp140 = tl.where(tmp67, tmp138, tmp139)
    tmp141 = tmp105 != tmp109
    tmp142 = tmp115 != tmp117
    tmp143 = tmp141 | tmp142
    tmp144 = tl.full(tmp143.shape, 0.0, tmp143.dtype)
    tmp145 = tl.where(tmp95, tmp143, tmp144)
    tmp146 = tl.where(tmp67, tmp140, tmp145)
    tmp147 = tl.where(tmp36, tmp135, tmp146)
    tmp148 = tl.where(tmp4, tmp130, tmp147)
    tmp150 = 0.0
    tmp151 = tl.where(tmp148, tmp150, tmp149)
    tmp152 = tmp125.to(tl.int64)
    tl.device_assert(((0 <= tmp152) & (tmp152 < 16)) | ~(xmask), "index out of bounds: 0 <= tmp152 < 16")
    tl.store(out_ptr3 + (x2), tmp151, xmask)
    tl.atomic_add(out_ptr4 + (tmp152 + 16*x1), tmp151, xmask, sem='relaxed')
