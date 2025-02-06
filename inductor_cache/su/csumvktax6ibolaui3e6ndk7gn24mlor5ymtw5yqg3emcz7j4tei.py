
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_leaky_relu_mul_sub_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_leaky_relu_mul_sub_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x6 = xindex // 16
    x2 = ((xindex // 16) % 512)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 2*tmp4 + 4*x6), None, eviction_policy='evict_last').to(tl.int1)
    tmp10 = tl.load(in_ptr3 + (tmp8 + 2*tmp4 + 4*x6), None, eviction_policy='evict_last')
    tmp12 = tmp10 + tmp11
    tmp13 = 0.1
    tmp14 = tmp12 * tmp13
    tmp15 = tl.where(tmp9, tmp12, tmp14)
    tmp17 = tmp16 + tmp1
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr2 + (tmp8 + 2*tmp19 + 4*x6), None, eviction_policy='evict_last').to(tl.int1)
    tmp21 = tl.load(in_ptr3 + (tmp8 + 2*tmp19 + 4*x6), None, eviction_policy='evict_last')
    tmp22 = tmp21 + tmp11
    tmp23 = tmp22 * tmp13
    tmp24 = tl.where(tmp20, tmp22, tmp23)
    tmp26 = tmp25 + tmp1
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr2 + (tmp28 + 2*tmp19 + 4*x6), None, eviction_policy='evict_last').to(tl.int1)
    tmp30 = tl.load(in_ptr3 + (tmp28 + 2*tmp19 + 4*x6), None, eviction_policy='evict_last')
    tmp31 = tmp30 + tmp11
    tmp32 = tmp31 * tmp13
    tmp33 = tl.where(tmp29, tmp31, tmp32)
    tmp34 = tmp33 - tmp24
    tmp36 = tmp34 * tmp35
    tmp37 = tmp24 + tmp36
    tmp38 = tl.load(in_ptr2 + (tmp28 + 2*tmp4 + 4*x6), None, eviction_policy='evict_last').to(tl.int1)
    tmp39 = tl.load(in_ptr3 + (tmp28 + 2*tmp4 + 4*x6), None, eviction_policy='evict_last')
    tmp40 = tmp39 + tmp11
    tmp41 = tmp40 * tmp13
    tmp42 = tl.where(tmp38, tmp40, tmp41)
    tmp43 = tmp42 - tmp15
    tmp44 = tmp43 * tmp35
    tmp45 = tmp15 + tmp44
    tmp46 = tmp45 - tmp37
    tmp48 = tmp46 * tmp47
    tmp49 = tmp37 + tmp48
    tl.store(in_out_ptr0 + (x4), tmp49, None)
