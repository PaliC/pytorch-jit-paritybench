
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 100)
    x0 = (xindex % 16)
    x2 = xindex // 1600
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 52, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (48*x0 + 768*x2 + ((-4) + x1)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-4) + x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 - tmp11
    tmp13 = tl.load(in_ptr3 + ((-4) + x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp22 = tl.load(in_ptr4 + ((-4) + x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + ((-4) + x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp9, tmp27, tmp28)
    tmp30 = tmp0 >= tmp7
    tmp31 = tl.full([1], 100, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tl.load(in_ptr6 + (48*x0 + 768*x2 + ((-52) + x1)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr7 + ((-52) + x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp33 - tmp34
    tmp36 = tl.load(in_ptr8 + ((-52) + x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = 1e-05
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.sqrt(tmp38)
    tmp40 = tl.full([1], 1, tl.int32)
    tmp41 = tmp40 / tmp39
    tmp42 = 1.0
    tmp43 = tmp41 * tmp42
    tmp44 = tmp35 * tmp43
    tmp45 = tl.load(in_ptr9 + ((-52) + x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp44 * tmp45
    tmp47 = tl.load(in_ptr10 + ((-52) + x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp46 + tmp47
    tmp49 = tl.full([1], 0, tl.int32)
    tmp50 = triton_helpers.maximum(tmp49, tmp48)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tl.where(tmp9, tmp29, tmp52)
    tmp54 = tl.where(tmp4, tmp5, tmp53)
    tl.store(out_ptr0 + (x3), tmp54, xmask)
