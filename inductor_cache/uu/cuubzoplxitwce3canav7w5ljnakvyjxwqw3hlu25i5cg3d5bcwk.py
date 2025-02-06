
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 77312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 19328)
    x1 = xindex // 19328
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12544, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = triton_helpers.div_floor_integer((((x0) // 14) % 14),  14)
    tmp6 = 1 + (triton_helpers.div_floor_integer((((x0) // 14) % 14),  14))
    tmp7 = tmp5 < tmp6
    tmp8 = triton_helpers.div_floor_integer(((x0) % 14),  14)
    tmp9 = 1 + (triton_helpers.div_floor_integer(((x0) % 14),  14))
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tmp11 & tmp4
    tmp13 = tl.load(in_ptr0 + (64*x1 + ((((x0) // 196) % 64))), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = 1.0
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = tmp13 / tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 18816, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tl.load(in_ptr1 + (6272*x1 + ((-12544) + x0)), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp0 >= tmp21
    tmp26 = tl.full([1], 19328, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr2 + (512*x1 + ((-18816) + x0)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr3 + ((-18816) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr4 + ((-18816) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr5 + ((-18816) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr6 + ((-18816) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = 0.0
    tmp45 = triton_helpers.maximum(tmp43, tmp44)
    tmp46 = 6.0
    tmp47 = triton_helpers.minimum(tmp45, tmp46)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp25, tmp47, tmp48)
    tmp50 = tl.where(tmp23, tmp24, tmp49)
    tmp51 = tl.where(tmp4, tmp19, tmp50)
    tl.store(out_ptr0 + (x2), tmp51, xmask)
