
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], -1, tl.int64)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5 & tmp5
    tmp7 = tl.load(in_ptr0 + ((-3) + 4*x0), tmp6 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp8 = tmp1 >= tmp1
    tmp9 = tmp1 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-2) + 4*x0), tmp11 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp13 = triton_helpers.maximum(tmp12, tmp7)
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-1) + 4*x0), tmp18 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp20 = triton_helpers.maximum(tmp19, tmp13)
    tmp21 = tmp10 & tmp5
    tmp22 = tl.load(in_ptr0 + ((-1) + 4*x0), tmp21 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp23 = triton_helpers.maximum(tmp22, tmp20)
    tmp24 = tmp10 & tmp10
    tmp25 = tl.load(in_ptr0 + (4*x0), tmp24 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp26 = triton_helpers.maximum(tmp25, tmp23)
    tmp27 = tmp10 & tmp17
    tmp28 = tl.load(in_ptr0 + (1 + 4*x0), tmp27 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp29 = triton_helpers.maximum(tmp28, tmp26)
    tmp30 = tmp17 & tmp5
    tmp31 = tl.load(in_ptr0 + (1 + 4*x0), tmp30 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp29)
    tmp33 = tmp17 & tmp10
    tmp34 = tl.load(in_ptr0 + (2 + 4*x0), tmp33 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp17 & tmp17
    tmp37 = tl.load(in_ptr0 + (3 + 4*x0), tmp36 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = tmp12 > tmp7
    tmp40 = tl.full([1], 1, tl.int8)
    tmp41 = tl.full([1], 0, tl.int8)
    tmp42 = tl.where(tmp39, tmp40, tmp41)
    tmp43 = tmp19 > tmp13
    tmp44 = tl.full([1], 2, tl.int8)
    tmp45 = tl.where(tmp43, tmp44, tmp42)
    tmp46 = tmp22 > tmp20
    tmp47 = tl.full([1], 3, tl.int8)
    tmp48 = tl.where(tmp46, tmp47, tmp45)
    tmp49 = tmp25 > tmp23
    tmp50 = tl.full([1], 4, tl.int8)
    tmp51 = tl.where(tmp49, tmp50, tmp48)
    tmp52 = tmp28 > tmp26
    tmp53 = tl.full([1], 5, tl.int8)
    tmp54 = tl.where(tmp52, tmp53, tmp51)
    tmp55 = tmp31 > tmp29
    tmp56 = tl.full([1], 6, tl.int8)
    tmp57 = tl.where(tmp55, tmp56, tmp54)
    tmp58 = tmp34 > tmp32
    tmp59 = tl.full([1], 7, tl.int8)
    tmp60 = tl.where(tmp58, tmp59, tmp57)
    tmp61 = tmp37 > tmp35
    tmp62 = tl.full([1], 8, tl.int8)
    tmp63 = tl.where(tmp61, tmp62, tmp60)
    tl.store(out_ptr0 + (x0), tmp38, xmask)
    tl.store(out_ptr1 + (x0), tmp63, xmask)
