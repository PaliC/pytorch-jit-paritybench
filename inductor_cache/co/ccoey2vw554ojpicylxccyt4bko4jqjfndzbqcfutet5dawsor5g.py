
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 28, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_26(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8192)
    x1 = xindex // 8192
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 4096, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.full([1], -2, tl.int64)
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 1, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp15
    tmp17 = tmp16 & tmp9
    tmp18 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp17, eviction_policy='evict_last', other=float("-inf"))
    tmp19 = tl.full([1], -1, tl.int64)
    tmp20 = tmp19 >= tmp11
    tmp21 = tmp19 < tmp13
    tmp22 = tmp20 & tmp21
    tmp23 = tmp15 & tmp22
    tmp24 = tmp23 & tmp9
    tmp25 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp24, eviction_policy='evict_last', other=float("-inf"))
    tmp26 = triton_helpers.maximum(tmp25, tmp18)
    tmp27 = tmp11 >= tmp11
    tmp28 = tmp11 < tmp13
    tmp29 = tmp27 & tmp28
    tmp30 = tmp15 & tmp29
    tmp31 = tmp30 & tmp9
    tmp32 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp31, eviction_policy='evict_last', other=float("-inf"))
    tmp33 = triton_helpers.maximum(tmp32, tmp26)
    tmp34 = tmp13 >= tmp11
    tmp35 = tmp13 < tmp13
    tmp36 = tmp34 & tmp35
    tmp37 = tmp15 & tmp36
    tmp38 = tmp37 & tmp9
    tmp39 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp38, eviction_policy='evict_last', other=float("-inf"))
    tmp40 = triton_helpers.maximum(tmp39, tmp33)
    tmp41 = tl.full([1], 2, tl.int64)
    tmp42 = tmp41 >= tmp11
    tmp43 = tmp41 < tmp13
    tmp44 = tmp42 & tmp43
    tmp45 = tmp15 & tmp44
    tmp46 = tmp45 & tmp9
    tmp47 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp46, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp40)
    tmp49 = tmp22 & tmp15
    tmp50 = tmp49 & tmp9
    tmp51 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp50, eviction_policy='evict_last', other=float("-inf"))
    tmp52 = triton_helpers.maximum(tmp51, tmp48)
    tmp53 = tmp22 & tmp22
    tmp54 = tmp53 & tmp9
    tmp55 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp54, eviction_policy='evict_last', other=float("-inf"))
    tmp56 = triton_helpers.maximum(tmp55, tmp52)
    tmp57 = tmp22 & tmp29
    tmp58 = tmp57 & tmp9
    tmp59 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp58, eviction_policy='evict_last', other=float("-inf"))
    tmp60 = triton_helpers.maximum(tmp59, tmp56)
    tmp61 = tmp22 & tmp36
    tmp62 = tmp61 & tmp9
    tmp63 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp62, eviction_policy='evict_last', other=float("-inf"))
    tmp64 = triton_helpers.maximum(tmp63, tmp60)
    tmp65 = tmp22 & tmp44
    tmp66 = tmp65 & tmp9
    tmp67 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp66, eviction_policy='evict_last', other=float("-inf"))
    tmp68 = triton_helpers.maximum(tmp67, tmp64)
    tmp69 = tmp29 & tmp15
    tmp70 = tmp69 & tmp9
    tmp71 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp70, eviction_policy='evict_last', other=float("-inf"))
    tmp72 = triton_helpers.maximum(tmp71, tmp68)
    tmp73 = tmp29 & tmp22
    tmp74 = tmp73 & tmp9
    tmp75 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp74, eviction_policy='evict_last', other=float("-inf"))
    tmp76 = triton_helpers.maximum(tmp75, tmp72)
    tmp77 = tmp29 & tmp29
    tmp78 = tmp77 & tmp9
    tmp79 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp78, eviction_policy='evict_last', other=float("-inf"))
    tmp80 = triton_helpers.maximum(tmp79, tmp76)
    tmp81 = tmp29 & tmp36
    tmp82 = tmp81 & tmp9
    tmp83 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp82, eviction_policy='evict_last', other=float("-inf"))
    tmp84 = triton_helpers.maximum(tmp83, tmp80)
    tmp85 = tmp29 & tmp44
    tmp86 = tmp85 & tmp9
    tmp87 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp86, eviction_policy='evict_last', other=float("-inf"))
    tmp88 = triton_helpers.maximum(tmp87, tmp84)
    tmp89 = tmp36 & tmp15
    tmp90 = tmp89 & tmp9
    tmp91 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp90, eviction_policy='evict_last', other=float("-inf"))
    tmp92 = triton_helpers.maximum(tmp91, tmp88)
    tmp93 = tmp36 & tmp22
    tmp94 = tmp93 & tmp9
    tmp95 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp94, eviction_policy='evict_last', other=float("-inf"))
    tmp96 = triton_helpers.maximum(tmp95, tmp92)
    tmp97 = tmp36 & tmp29
    tmp98 = tmp97 & tmp9
    tmp99 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp98, eviction_policy='evict_last', other=float("-inf"))
    tmp100 = triton_helpers.maximum(tmp99, tmp96)
    tmp101 = tmp36 & tmp36
    tmp102 = tmp101 & tmp9
    tmp103 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp102, eviction_policy='evict_last', other=float("-inf"))
    tmp104 = triton_helpers.maximum(tmp103, tmp100)
    tmp105 = tmp36 & tmp44
    tmp106 = tmp105 & tmp9
    tmp107 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp106, eviction_policy='evict_last', other=float("-inf"))
    tmp108 = triton_helpers.maximum(tmp107, tmp104)
    tmp109 = tmp44 & tmp15
    tmp110 = tmp109 & tmp9
    tmp111 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp110, eviction_policy='evict_last', other=float("-inf"))
    tmp112 = triton_helpers.maximum(tmp111, tmp108)
    tmp113 = tmp44 & tmp22
    tmp114 = tmp113 & tmp9
    tmp115 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp114, eviction_policy='evict_last', other=float("-inf"))
    tmp116 = triton_helpers.maximum(tmp115, tmp112)
    tmp117 = tmp44 & tmp29
    tmp118 = tmp117 & tmp9
    tmp119 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp118, eviction_policy='evict_last', other=float("-inf"))
    tmp120 = triton_helpers.maximum(tmp119, tmp116)
    tmp121 = tmp44 & tmp36
    tmp122 = tmp121 & tmp9
    tmp123 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp122, eviction_policy='evict_last', other=float("-inf"))
    tmp124 = triton_helpers.maximum(tmp123, tmp120)
    tmp125 = tmp44 & tmp44
    tmp126 = tmp125 & tmp9
    tmp127 = tl.load(in_ptr0 + (2048*x1 + ((-2048) + x0)), tmp126, eviction_policy='evict_last', other=float("-inf"))
    tmp128 = triton_helpers.maximum(tmp127, tmp124)
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp9, tmp128, tmp129)
    tmp131 = tmp0 >= tmp7
    tmp132 = tl.full([1], 6144, tl.int64)
    tmp133 = tmp0 < tmp132
    tmp134 = tmp131 & tmp133
    tmp135 = tl.load(in_ptr1 + (2048*x1 + ((-4096) + x0)), tmp134, eviction_policy='evict_last', other=0.0)
    tmp136 = tmp0 >= tmp132
    tmp137 = tl.full([1], 8192, tl.int64)
    tmp138 = tmp0 < tmp137
    tmp139 = tl.load(in_ptr2 + (2048*x1 + ((-6144) + x0)), tmp136, eviction_policy='evict_last', other=0.0)
    tmp140 = tl.where(tmp134, tmp135, tmp139)
    tmp141 = tl.where(tmp9, tmp130, tmp140)
    tmp142 = tl.where(tmp4, tmp5, tmp141)
    tl.store(out_ptr0 + (x2), tmp142, None)
