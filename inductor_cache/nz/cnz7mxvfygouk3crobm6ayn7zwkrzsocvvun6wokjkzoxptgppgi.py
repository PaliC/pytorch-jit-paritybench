
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 32)
    x0 = (xindex % 256)
    x2 = xindex // 8192
    x3 = xindex
    tmp41 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 2048*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 16, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 256*((-8) + x1) + 2048*x2), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr0 + (x0 + 256*((-8) + x1) + 2048*x2), tmp9, other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tmp0 >= tmp7
    tmp16 = tl.full([1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + (x0 + 256*((-16) + x1) + 2048*x2), tmp18, other=0.0)
    tmp20 = tl.load(in_ptr1 + (x0 + 256*((-16) + x1) + 2048*x2), tmp18, other=0.0)
    tmp21 = tl.load(in_ptr0 + (x0 + 256*((-16) + x1) + 2048*x2), tmp18, other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp18, tmp23, tmp24)
    tmp26 = tmp0 >= tmp16
    tmp27 = tl.full([1], 32, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.load(in_ptr3 + (x0 + 256*((-24) + x1) + 2048*x2), tmp26, other=0.0)
    tmp30 = tl.load(in_ptr2 + (x0 + 256*((-24) + x1) + 2048*x2), tmp26, other=0.0)
    tmp31 = tl.load(in_ptr1 + (x0 + 256*((-24) + x1) + 2048*x2), tmp26, other=0.0)
    tmp32 = tl.load(in_ptr0 + (x0 + 256*((-24) + x1) + 2048*x2), tmp26, other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tmp30 + tmp33
    tmp35 = tmp29 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp26, tmp35, tmp36)
    tmp38 = tl.where(tmp18, tmp25, tmp37)
    tmp39 = tl.where(tmp9, tmp14, tmp38)
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
    tmp56 = 0.0
    tmp57 = tmp55 > tmp56
    tmp59 = tmp58 * tmp55
    tmp60 = tl.where(tmp57, tmp55, tmp59)
    tl.store(out_ptr0 + (x3), tmp40, None)
    tl.store(in_out_ptr0 + (x3), tmp60, None)
