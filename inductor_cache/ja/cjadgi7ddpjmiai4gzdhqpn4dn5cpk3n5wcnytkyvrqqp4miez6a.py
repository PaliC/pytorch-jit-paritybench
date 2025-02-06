
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
    triton_meta={'signature': {'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_out_ptr7': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_constant_pad_nd_mul_sub_0', 'mutated_arg_names': ['in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6', 'in_out_ptr7'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_constant_pad_nd_mul_sub_0(in_out_ptr4, in_out_ptr5, in_out_ptr6, in_out_ptr7, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 2, tl.int64)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr0 + (10 + 16*x0), tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp9 >= tmp1
    tmp11 = tmp9 < tmp3
    tmp12 = tmp5 & tmp10
    tmp13 = tmp12 & tmp11
    tmp14 = tl.load(in_ptr0 + (9 + 16*x0), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp8 - tmp14
    tmp16 = 0.5
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tmp10 & tmp11
    tmp20 = tmp19 & tmp2
    tmp21 = tmp20 & tmp4
    tmp22 = tl.load(in_ptr0 + (6 + 16*x0), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp19 & tmp10
    tmp24 = tmp23 & tmp11
    tmp25 = tl.load(in_ptr0 + (5 + 16*x0), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp22 - tmp25
    tmp27 = tmp26 * tmp16
    tmp28 = tmp25 + tmp27
    tmp29 = tmp18 - tmp28
    tmp30 = tmp29 * tmp16
    tmp31 = tmp28 + tmp30
    tl.store(in_out_ptr4 + (x0), tmp31, xmask)
    tl.store(in_out_ptr5 + (x0), tmp31, xmask)
    tl.store(in_out_ptr6 + (x0), tmp31, xmask)
    tl.store(in_out_ptr7 + (x0), tmp31, xmask)
