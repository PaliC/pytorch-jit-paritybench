
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_79', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_79(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 704)
    x0 = (xindex % 4)
    x2 = xindex // 2816
    x3 = xindex
    tmp41 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 2048*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 544, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4*((-512) + x1) + 128*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 576, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 4*((-544) + x1) + 128*x2), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 608, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + 4*((-576) + x1) + 128*x2), tmp19 & xmask, other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 640, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x0 + 4*((-608) + x1) + 128*x2), tmp24 & xmask, other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 672, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x0 + 4*((-640) + x1) + 128*x2), tmp29 & xmask, other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 704, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr6 + (x0 + 4*((-672) + x1) + 128*x2), tmp31 & xmask, other=0.0)
    tmp35 = tl.where(tmp29, tmp30, tmp34)
    tmp36 = tl.where(tmp24, tmp25, tmp35)
    tmp37 = tl.where(tmp19, tmp20, tmp36)
    tmp38 = tl.where(tmp14, tmp15, tmp37)
    tmp39 = tl.where(tmp9, tmp10, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tmp42 = tmp40 - tmp41
    tmp44 = 1e-05
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.sqrt(tmp45)
    tmp47 = tl.full([1], 1, tl.int32)
    tmp48 = tmp47 / tmp46
    tmp49 = 1.0
    tmp50 = tmp48 * tmp49
    tmp51 = tmp42 * tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tl.full([1], 0, tl.int32)
    tmp57 = triton_helpers.maximum(tmp56, tmp55)
    tl.store(out_ptr0 + (x3), tmp40, xmask)
    tl.store(out_ptr1 + (x3), tmp57, xmask)
