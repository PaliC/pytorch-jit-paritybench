
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_relu_sub_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_mul_relu_sub_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x5 = xindex // 16
    x2 = ((xindex // 16) % 256)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr9 + (x6), None)
    tmp46 = tl.load(in_ptr10 + (x2), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr11 + (x6), None)
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 2*tmp4 + 4*x5), None, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr4 + (tmp8 + 2*tmp4 + 4*x5), None, eviction_policy='evict_last')
    tmp13 = tmp11 + tmp12
    tmp15 = tmp14 + tmp1
    tmp16 = tmp14 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp14)
    tmp18 = tl.load(in_ptr2 + (tmp8 + 2*tmp17 + 4*x5), None, eviction_policy='evict_last')
    tmp19 = tmp18 + tmp10
    tmp20 = tl.load(in_ptr4 + (tmp8 + 2*tmp17 + 4*x5), None, eviction_policy='evict_last')
    tmp21 = tmp19 + tmp20
    tmp23 = tmp22 + tmp1
    tmp24 = tmp22 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp22)
    tmp26 = tl.load(in_ptr2 + (tmp25 + 2*tmp17 + 4*x5), None, eviction_policy='evict_last')
    tmp27 = tmp26 + tmp10
    tmp28 = tl.load(in_ptr4 + (tmp25 + 2*tmp17 + 4*x5), None, eviction_policy='evict_last')
    tmp29 = tmp27 + tmp28
    tmp30 = tmp29 - tmp21
    tmp32 = tmp30 * tmp31
    tmp33 = tmp21 + tmp32
    tmp34 = tl.load(in_ptr2 + (tmp25 + 2*tmp4 + 4*x5), None, eviction_policy='evict_last')
    tmp35 = tmp34 + tmp10
    tmp36 = tl.load(in_ptr4 + (tmp25 + 2*tmp4 + 4*x5), None, eviction_policy='evict_last')
    tmp37 = tmp35 + tmp36
    tmp38 = tmp37 - tmp13
    tmp39 = tmp38 * tmp31
    tmp40 = tmp13 + tmp39
    tmp41 = tmp40 - tmp33
    tmp43 = tmp41 * tmp42
    tmp44 = tmp33 + tmp43
    tmp47 = tmp45 + tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tmp44 + tmp49
    tmp51 = tl.full([1], 0, tl.int32)
    tmp52 = triton_helpers.maximum(tmp51, tmp50)
    tl.store(in_out_ptr0 + (x6), tmp52, None)
