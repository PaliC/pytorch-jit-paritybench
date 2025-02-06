
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (12 + x0 + 16*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (x0 + 16*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr0 + (4 + x0 + 16*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.load(in_ptr0 + (8 + x0 + 16*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = 3.0
    tmp16 = tmp14 / tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp9, tmp16, tmp17)
    tmp19 = tmp0 >= tmp7
    tmp20 = tl.full([1], 3, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tl.load(in_ptr0 + (x0 + 16*x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr0 + (4 + x0 + 16*x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp26 = tl.load(in_ptr0 + (8 + x0 + 16*x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp22, tmp27, tmp28)
    tmp30 = tmp0 >= tmp20
    tmp31 = tl.full([1], 4, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tl.load(in_ptr1 + (3*x2), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp33 - tmp33
    tmp35 = tl_math.exp(tmp34)
    tmp36 = tmp35 / tmp35
    tmp37 = tl.load(in_ptr0 + (x0 + 16*x2), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp36 * tmp37
    tmp39 = tl.load(in_ptr1 + (1 + 3*x2), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp39 - tmp39
    tmp41 = tl_math.exp(tmp40)
    tmp42 = tmp41 / tmp41
    tmp43 = tl.load(in_ptr0 + (4 + x0 + 16*x2), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 * tmp43
    tmp45 = tmp38 + tmp44
    tmp46 = tl.load(in_ptr1 + (2 + 3*x2), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp46 - tmp46
    tmp48 = tl_math.exp(tmp47)
    tmp49 = tmp48 / tmp48
    tmp50 = tl.load(in_ptr0 + (8 + x0 + 16*x2), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp49 * tmp50
    tmp52 = tmp45 + tmp51
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp30, tmp52, tmp53)
    tmp55 = tl.where(tmp22, tmp29, tmp54)
    tmp56 = tl.where(tmp9, tmp18, tmp55)
    tmp57 = tl.where(tmp4, tmp5, tmp56)
    tl.store(out_ptr0 + (x4), tmp57, xmask)
