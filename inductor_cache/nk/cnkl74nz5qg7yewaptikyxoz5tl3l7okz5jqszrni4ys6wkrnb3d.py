
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_add_ones_scalar_tensor_tril_where_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp1 = (-1)*x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 <= tmp2
    tmp4 = tl.full([1], True, tl.int1)
    tmp5 = tmp3 & tmp4
    tmp6 = 0.0
    tmp7 = float("-inf")
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp0 + tmp8
    tmp11 = 1 + ((-1)*x0)
    tmp12 = tmp11 <= tmp2
    tmp13 = tmp12 & tmp4
    tmp14 = tl.where(tmp13, tmp6, tmp7)
    tmp15 = tmp10 + tmp14
    tmp16 = triton_helpers.maximum(tmp9, tmp15)
    tmp18 = 2 + ((-1)*x0)
    tmp19 = tmp18 <= tmp2
    tmp20 = tmp19 & tmp4
    tmp21 = tl.where(tmp20, tmp6, tmp7)
    tmp22 = tmp17 + tmp21
    tmp23 = triton_helpers.maximum(tmp16, tmp22)
    tmp25 = 3 + ((-1)*x0)
    tmp26 = tmp25 <= tmp2
    tmp27 = tmp26 & tmp4
    tmp28 = tl.where(tmp27, tmp6, tmp7)
    tmp29 = tmp24 + tmp28
    tmp30 = triton_helpers.maximum(tmp23, tmp29)
    tmp31 = tmp9 - tmp30
    tmp32 = tl_math.exp(tmp31)
    tmp33 = tmp15 - tmp30
    tmp34 = tl_math.exp(tmp33)
    tmp35 = tmp32 + tmp34
    tmp36 = tmp22 - tmp30
    tmp37 = tl_math.exp(tmp36)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp29 - tmp30
    tmp40 = tl_math.exp(tmp39)
    tmp41 = tmp38 + tmp40
    tmp42 = tmp9 == tmp7
    tmp43 = tmp42 == 0
    tmp44 = tmp43.to(tl.int64)
    tmp45 = (tmp44 != 0)
    tmp46 = tmp15 == tmp7
    tmp47 = tmp46 == 0
    tmp48 = tmp47.to(tl.int64)
    tmp49 = (tmp48 != 0)
    tmp50 = tmp45 | tmp49
    tmp51 = tmp22 == tmp7
    tmp52 = tmp51 == 0
    tmp53 = tmp52.to(tl.int64)
    tmp54 = (tmp53 != 0)
    tmp55 = tmp50 | tmp54
    tmp56 = tmp29 == tmp7
    tmp57 = tmp56 == 0
    tmp58 = tmp57.to(tl.int64)
    tmp59 = (tmp58 != 0)
    tmp60 = tmp55 | tmp59
    tl.store(out_ptr0 + (x2), tmp30, xmask)
    tl.store(out_ptr1 + (x2), tmp41, xmask)
    tl.store(out_ptr2 + (x2), tmp60, xmask)
