
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp20 = triton_helpers.maximum(tmp19, tmp18)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp28 = triton_helpers.maximum(tmp27, tmp26)
    tmp30 = triton_helpers.maximum(tmp29, tmp28)
    tmp31 = tmp1 > tmp0
    tmp32 = tl.full([1], 1, tl.int8)
    tmp33 = tl.full([1], 0, tl.int8)
    tmp34 = tl.where(tmp31, tmp32, tmp33)
    tmp35 = tmp3 > tmp2
    tmp36 = tl.full([1], 2, tl.int8)
    tmp37 = tl.where(tmp35, tmp36, tmp34)
    tmp38 = tmp5 > tmp4
    tmp39 = tl.full([1], 3, tl.int8)
    tmp40 = tl.where(tmp38, tmp39, tmp37)
    tmp41 = tmp7 > tmp6
    tmp42 = tl.full([1], 4, tl.int8)
    tmp43 = tl.where(tmp41, tmp42, tmp40)
    tmp44 = tmp9 > tmp8
    tmp45 = tl.full([1], 5, tl.int8)
    tmp46 = tl.where(tmp44, tmp45, tmp43)
    tmp47 = tmp11 > tmp10
    tmp48 = tl.full([1], 6, tl.int8)
    tmp49 = tl.where(tmp47, tmp48, tmp46)
    tmp50 = tmp13 > tmp12
    tmp51 = tl.full([1], 7, tl.int8)
    tmp52 = tl.where(tmp50, tmp51, tmp49)
    tmp53 = tmp15 > tmp14
    tmp54 = tl.full([1], 8, tl.int8)
    tmp55 = tl.where(tmp53, tmp54, tmp52)
    tmp56 = tmp17 > tmp16
    tmp57 = tl.full([1], 9, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp19 > tmp18
    tmp60 = tl.full([1], 10, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp21 > tmp20
    tmp63 = tl.full([1], 11, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp23 > tmp22
    tmp66 = tl.full([1], 12, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp25 > tmp24
    tmp69 = tl.full([1], 13, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp27 > tmp26
    tmp72 = tl.full([1], 14, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp29 > tmp28
    tmp75 = tl.full([1], 15, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tl.full([1], 4, tl.int32)
    tmp78 = tl.where((tmp76 < 0) != (tmp77 < 0), tl.where(tmp76 % tmp77 != 0, tmp76 // tmp77 - 1, tmp76 // tmp77), tmp76 // tmp77)
    tmp79 = tmp78 * tmp77
    tmp80 = tmp76 - tmp79
    tmp81 = tl.full([1], 0, tl.int64)
    tmp82 = tmp81 + tmp78
    tmp83 = tmp81 + tmp80
    tmp84 = tl.full([1], 4, tl.int64)
    tmp85 = tmp82 * tmp84
    tmp86 = tmp85 + tmp83
    tl.store(out_ptr0 + (x0), tmp30, xmask)
    tl.store(out_ptr2 + (x0), tmp86, xmask)
