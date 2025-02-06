
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_max_pool2d_with_indices_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_max_pool2d_with_indices_11(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 33) % 33)
    x0 = (xindex % 33)
    x2 = xindex // 1089
    x4 = (xindex % 1089)
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 65, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-66) + 2*x0 + 130*x1 + 4256*x2), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-65) + 2*x0 + 130*x1 + 4256*x2), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-64) + 2*x0 + 130*x1 + 4256*x2), tmp23 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 130*x1 + 4256*x2), tmp30 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 130*x1 + 4256*x2), tmp33 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 130*x1 + 4256*x2), tmp36 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (64 + 2*x0 + 130*x1 + 4256*x2), tmp43 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (65 + 2*x0 + 130*x1 + 4256*x2), tmp46 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (66 + 2*x0 + 130*x1 + 4256*x2), tmp49 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp17 > tmp11
    tmp53 = tl.full([1], 1, tl.int8)
    tmp54 = tl.full([1], 0, tl.int8)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp24 > tmp18
    tmp57 = tl.full([1], 2, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp31 > tmp25
    tmp60 = tl.full([1], 3, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp32
    tmp63 = tl.full([1], 4, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 5, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp44 > tmp38
    tmp69 = tl.full([1], 6, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp45
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 8, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tl.load(in_ptr0 + ((-66) + 2*x0 + 130*x1 + 4256*x2), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tl.load(in_ptr0 + ((-65) + 2*x0 + 130*x1 + 4256*x2), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 + tmp77
    tmp80 = tl.load(in_ptr0 + ((-64) + 2*x0 + 130*x1 + 4256*x2), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp81 = tmp80 + tmp79
    tmp82 = tl.load(in_ptr0 + ((-1) + 2*x0 + 130*x1 + 4256*x2), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tmp82 + tmp81
    tmp84 = tl.load(in_ptr0 + (2*x0 + 130*x1 + 4256*x2), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp84 + tmp83
    tmp86 = tl.load(in_ptr0 + (1 + 2*x0 + 130*x1 + 4256*x2), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp86 + tmp85
    tmp88 = tl.load(in_ptr0 + (64 + 2*x0 + 130*x1 + 4256*x2), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp88 + tmp87
    tmp90 = tl.load(in_ptr0 + (65 + 2*x0 + 130*x1 + 4256*x2), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp90 + tmp89
    tmp92 = tl.load(in_ptr0 + (66 + 2*x0 + 130*x1 + 4256*x2), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp93 = tmp92 + tmp91
    tmp94 = ((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))*((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0))) + ((65) * ((65) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (65)))*((65) * ((65) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (65))) + ((-1)*((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))*((65) * ((65) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (65)))) + ((-1)*((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0)))*((65) * ((65) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (65))))
    tmp95 = tmp93 / tmp94
    tl.store(out_ptr0 + (x4 + 1120*x2), tmp51, xmask)
    tl.store(out_ptr1 + (x4 + 1152*x2), tmp76, xmask)
    tl.store(out_ptr2 + (x4 + 1120*x2), tmp95, xmask)
