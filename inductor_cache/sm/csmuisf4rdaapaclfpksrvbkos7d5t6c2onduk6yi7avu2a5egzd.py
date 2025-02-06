
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 2) % 2)
    x0 = (xindex % 2)
    x2 = xindex // 4
    x4 = xindex
    tmp0 = 4*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 5, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 4*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (4*x0 + 20*x1 + 25*x2), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 1 + 4*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (1 + 4*x0 + 20*x1 + 25*x2), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 2 + 4*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + (2 + 4*x0 + 20*x1 + 25*x2), tmp23 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 3 + 4*x0
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp5 & tmp29
    tmp31 = tl.load(in_ptr0 + (3 + 4*x0 + 20*x1 + 25*x2), tmp30 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = 1 + 4*x1
    tmp34 = tmp33 >= tmp1
    tmp35 = tmp33 < tmp3
    tmp36 = tmp34 & tmp35
    tmp37 = tmp36 & tmp9
    tmp38 = tl.load(in_ptr0 + (5 + 4*x0 + 20*x1 + 25*x2), tmp37 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp39 = triton_helpers.maximum(tmp38, tmp32)
    tmp40 = tmp36 & tmp15
    tmp41 = tl.load(in_ptr0 + (6 + 4*x0 + 20*x1 + 25*x2), tmp40 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp42 = triton_helpers.maximum(tmp41, tmp39)
    tmp43 = tmp36 & tmp22
    tmp44 = tl.load(in_ptr0 + (7 + 4*x0 + 20*x1 + 25*x2), tmp43 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp42)
    tmp46 = tmp36 & tmp29
    tmp47 = tl.load(in_ptr0 + (8 + 4*x0 + 20*x1 + 25*x2), tmp46 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = 2 + 4*x1
    tmp50 = tmp49 >= tmp1
    tmp51 = tmp49 < tmp3
    tmp52 = tmp50 & tmp51
    tmp53 = tmp52 & tmp9
    tmp54 = tl.load(in_ptr0 + (10 + 4*x0 + 20*x1 + 25*x2), tmp53 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp55 = triton_helpers.maximum(tmp54, tmp48)
    tmp56 = tmp52 & tmp15
    tmp57 = tl.load(in_ptr0 + (11 + 4*x0 + 20*x1 + 25*x2), tmp56 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp58 = triton_helpers.maximum(tmp57, tmp55)
    tmp59 = tmp52 & tmp22
    tmp60 = tl.load(in_ptr0 + (12 + 4*x0 + 20*x1 + 25*x2), tmp59 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp61 = triton_helpers.maximum(tmp60, tmp58)
    tmp62 = tmp52 & tmp29
    tmp63 = tl.load(in_ptr0 + (13 + 4*x0 + 20*x1 + 25*x2), tmp62 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp64 = triton_helpers.maximum(tmp63, tmp61)
    tmp65 = 3 + 4*x1
    tmp66 = tmp65 >= tmp1
    tmp67 = tmp65 < tmp3
    tmp68 = tmp66 & tmp67
    tmp69 = tmp68 & tmp9
    tmp70 = tl.load(in_ptr0 + (15 + 4*x0 + 20*x1 + 25*x2), tmp69 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp71 = triton_helpers.maximum(tmp70, tmp64)
    tmp72 = tmp68 & tmp15
    tmp73 = tl.load(in_ptr0 + (16 + 4*x0 + 20*x1 + 25*x2), tmp72 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp74 = triton_helpers.maximum(tmp73, tmp71)
    tmp75 = tmp68 & tmp22
    tmp76 = tl.load(in_ptr0 + (17 + 4*x0 + 20*x1 + 25*x2), tmp75 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp77 = triton_helpers.maximum(tmp76, tmp74)
    tmp78 = tmp68 & tmp29
    tmp79 = tl.load(in_ptr0 + (18 + 4*x0 + 20*x1 + 25*x2), tmp78 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp80 = triton_helpers.maximum(tmp79, tmp77)
    tmp81 = tmp17 > tmp11
    tmp82 = tl.full([1], 1, tl.int8)
    tmp83 = tl.full([1], 0, tl.int8)
    tmp84 = tl.where(tmp81, tmp82, tmp83)
    tmp85 = tmp24 > tmp18
    tmp86 = tl.full([1], 2, tl.int8)
    tmp87 = tl.where(tmp85, tmp86, tmp84)
    tmp88 = tmp31 > tmp25
    tmp89 = tl.full([1], 3, tl.int8)
    tmp90 = tl.where(tmp88, tmp89, tmp87)
    tmp91 = tmp38 > tmp32
    tmp92 = tl.full([1], 4, tl.int8)
    tmp93 = tl.where(tmp91, tmp92, tmp90)
    tmp94 = tmp41 > tmp39
    tmp95 = tl.full([1], 5, tl.int8)
    tmp96 = tl.where(tmp94, tmp95, tmp93)
    tmp97 = tmp44 > tmp42
    tmp98 = tl.full([1], 6, tl.int8)
    tmp99 = tl.where(tmp97, tmp98, tmp96)
    tmp100 = tmp47 > tmp45
    tmp101 = tl.full([1], 7, tl.int8)
    tmp102 = tl.where(tmp100, tmp101, tmp99)
    tmp103 = tmp54 > tmp48
    tmp104 = tl.full([1], 8, tl.int8)
    tmp105 = tl.where(tmp103, tmp104, tmp102)
    tmp106 = tmp57 > tmp55
    tmp107 = tl.full([1], 9, tl.int8)
    tmp108 = tl.where(tmp106, tmp107, tmp105)
    tmp109 = tmp60 > tmp58
    tmp110 = tl.full([1], 10, tl.int8)
    tmp111 = tl.where(tmp109, tmp110, tmp108)
    tmp112 = tmp63 > tmp61
    tmp113 = tl.full([1], 11, tl.int8)
    tmp114 = tl.where(tmp112, tmp113, tmp111)
    tmp115 = tmp70 > tmp64
    tmp116 = tl.full([1], 12, tl.int8)
    tmp117 = tl.where(tmp115, tmp116, tmp114)
    tmp118 = tmp73 > tmp71
    tmp119 = tl.full([1], 13, tl.int8)
    tmp120 = tl.where(tmp118, tmp119, tmp117)
    tmp121 = tmp76 > tmp74
    tmp122 = tl.full([1], 14, tl.int8)
    tmp123 = tl.where(tmp121, tmp122, tmp120)
    tmp124 = tmp79 > tmp77
    tmp125 = tl.full([1], 15, tl.int8)
    tmp126 = tl.where(tmp124, tmp125, tmp123)
    tl.store(out_ptr0 + (x4), tmp80, xmask)
    tl.store(out_ptr1 + (x4), tmp126, xmask)
