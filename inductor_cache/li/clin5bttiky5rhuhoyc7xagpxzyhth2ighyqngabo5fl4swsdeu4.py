
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], -1, tl.int64)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5 & tmp5
    tmp7 = tl.load(in_ptr0 + ((-5) + 16*x0), tmp6 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp8 = tmp1 >= tmp1
    tmp9 = tmp1 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-4) + 16*x0), tmp11 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp13 = triton_helpers.maximum(tmp12, tmp7)
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-3) + 16*x0), tmp18 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp20 = triton_helpers.maximum(tmp19, tmp13)
    tmp21 = tl.full([1], 2, tl.int64)
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + ((-2) + 16*x0), tmp25 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp27 = triton_helpers.maximum(tmp26, tmp20)
    tmp28 = tmp10 & tmp5
    tmp29 = tl.load(in_ptr0 + ((-1) + 16*x0), tmp28 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp30 = triton_helpers.maximum(tmp29, tmp27)
    tmp31 = tmp10 & tmp10
    tmp32 = tl.load(in_ptr0 + (16*x0), tmp31 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp33 = triton_helpers.maximum(tmp32, tmp30)
    tmp34 = tmp10 & tmp17
    tmp35 = tl.load(in_ptr0 + (1 + 16*x0), tmp34 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp36 = triton_helpers.maximum(tmp35, tmp33)
    tmp37 = tmp10 & tmp24
    tmp38 = tl.load(in_ptr0 + (2 + 16*x0), tmp37 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp39 = triton_helpers.maximum(tmp38, tmp36)
    tmp40 = tmp17 & tmp5
    tmp41 = tl.load(in_ptr0 + (3 + 16*x0), tmp40 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp42 = triton_helpers.maximum(tmp41, tmp39)
    tmp43 = tmp17 & tmp10
    tmp44 = tl.load(in_ptr0 + (4 + 16*x0), tmp43 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp42)
    tmp46 = tmp17 & tmp17
    tmp47 = tl.load(in_ptr0 + (5 + 16*x0), tmp46 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp17 & tmp24
    tmp50 = tl.load(in_ptr0 + (6 + 16*x0), tmp49 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp24 & tmp5
    tmp53 = tl.load(in_ptr0 + (7 + 16*x0), tmp52 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp54 = triton_helpers.maximum(tmp53, tmp51)
    tmp55 = tmp24 & tmp10
    tmp56 = tl.load(in_ptr0 + (8 + 16*x0), tmp55 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp57 = triton_helpers.maximum(tmp56, tmp54)
    tmp58 = tmp24 & tmp17
    tmp59 = tl.load(in_ptr0 + (9 + 16*x0), tmp58 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp60 = triton_helpers.maximum(tmp59, tmp57)
    tmp61 = tmp24 & tmp24
    tmp62 = tl.load(in_ptr0 + (10 + 16*x0), tmp61 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp63 = triton_helpers.maximum(tmp62, tmp60)
    tl.store(out_ptr0 + (x0), tmp63, xmask)
