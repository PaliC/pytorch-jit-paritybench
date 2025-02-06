
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_0(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_out_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr3 + (x3), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr14 + (x1), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr15 + (x1), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr16 + (x1), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr17 + (x1), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr18 + (x1), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr19 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp2 - tmp12
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = tl.full([1], 1, tl.int32)
    tmp19 = tmp18 / tmp17
    tmp20 = 1.0
    tmp21 = tmp19 * tmp20
    tmp22 = tmp13 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full([1], 0, tl.int32)
    tmp28 = triton_helpers.maximum(tmp27, tmp26)
    tmp30 = tmp5 - tmp29
    tmp32 = tmp31 + tmp15
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tmp18 / tmp33
    tmp35 = tmp34 * tmp20
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = triton_helpers.maximum(tmp27, tmp40)
    tmp42 = tmp28 + tmp41
    tmp44 = tmp8 - tmp43
    tmp46 = tmp45 + tmp15
    tmp47 = libdevice.sqrt(tmp46)
    tmp48 = tmp18 / tmp47
    tmp49 = tmp48 * tmp20
    tmp50 = tmp44 * tmp49
    tmp52 = tmp50 * tmp51
    tmp54 = tmp52 + tmp53
    tmp55 = triton_helpers.maximum(tmp27, tmp54)
    tmp56 = tmp42 + tmp55
    tmp58 = tmp11 - tmp57
    tmp60 = tmp59 + tmp15
    tmp61 = libdevice.sqrt(tmp60)
    tmp62 = tmp18 / tmp61
    tmp63 = tmp62 * tmp20
    tmp64 = tmp58 * tmp63
    tmp66 = tmp64 * tmp65
    tmp68 = tmp66 + tmp67
    tmp69 = triton_helpers.maximum(tmp27, tmp68)
    tmp70 = tmp56 + tmp69
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(in_out_ptr1 + (x3), tmp5, xmask)
    tl.store(in_out_ptr2 + (x3), tmp8, xmask)
    tl.store(in_out_ptr3 + (x3), tmp11, xmask)
    tl.store(in_out_ptr4 + (x3), tmp70, xmask)
