
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_convolution_relu_threshold_backward_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_convolution_relu_threshold_backward_3(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 6)
    x0 = (xindex % 16)
    x2 = xindex // 96
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr14 + (x1), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr15 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x1
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tl.load(in_ptr1 + (x0 + 16*x2), tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp3 >= tmp6
    tmp10 = tl.full([1], 3, tl.int64)
    tmp11 = tmp3 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + (x0 + 16*((-1) + x1) + 32*x2), tmp12 & xmask, other=0.0)
    tmp14 = tmp3 >= tmp10
    tmp15 = tl.full([1], 6, tl.int64)
    tmp16 = tmp3 < tmp15
    tmp17 = tl.load(in_ptr3 + (x0 + 16*((-3) + x1) + 48*x2), tmp14 & xmask, other=0.0)
    tmp18 = tl.load(in_ptr4 + ((-3) + x1), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 - tmp18
    tmp20 = tl.load(in_ptr5 + ((-3) + x1), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tl.full([1], 1, tl.int32)
    tmp25 = tmp24 / tmp23
    tmp26 = 1.0
    tmp27 = tmp25 * tmp26
    tmp28 = tmp19 * tmp27
    tmp29 = tl.load(in_ptr6 + ((-3) + x1), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.load(in_ptr7 + ((-3) + x1), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 + tmp31
    tmp33 = tl.full([1], 0, tl.int32)
    tmp34 = triton_helpers.maximum(tmp33, tmp32)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp14, tmp34, tmp35)
    tmp37 = tl.where(tmp12, tmp13, tmp36)
    tmp38 = tl.where(tmp7, tmp8, tmp37)
    tmp40 = tmp38 - tmp39
    tmp42 = 1e-05
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.sqrt(tmp43)
    tmp45 = tl.full([1], 1, tl.int32)
    tmp46 = tmp45 / tmp44
    tmp47 = 1.0
    tmp48 = tmp46 * tmp47
    tmp49 = tmp40 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tmp55 = tmp2 - tmp54
    tmp57 = tmp56 + tmp42
    tmp58 = libdevice.sqrt(tmp57)
    tmp59 = tmp45 / tmp58
    tmp60 = tmp59 * tmp47
    tmp61 = tmp55 * tmp60
    tmp63 = tmp61 * tmp62
    tmp65 = tmp63 + tmp64
    tmp66 = tmp53 + tmp65
    tmp67 = tl.full([1], 0, tl.int32)
    tmp68 = triton_helpers.maximum(tmp67, tmp66)
    tmp69 = 0.0
    tmp70 = tmp68 <= tmp69
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp38, xmask)
    tl.store(in_out_ptr1 + (x3), tmp68, xmask)
    tl.store(out_ptr1 + (x3), tmp70, xmask)
