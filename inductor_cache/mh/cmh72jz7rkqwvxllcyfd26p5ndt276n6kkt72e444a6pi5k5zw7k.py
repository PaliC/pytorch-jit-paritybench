
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 1192)
    x0 = (xindex % 16)
    x2 = xindex // 19072
    x3 = xindex
    tmp45 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 18304*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x0 + 16*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1192, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-1024) + x1
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.full([1], 144, tl.int64)
    tmp19 = tmp15 < tmp18
    tmp20 = tmp19 & tmp12
    tmp21 = (-1024) + x1
    tmp22 = tl.full([1], 0, tl.int64)
    tmp23 = tmp21 >= tmp22
    tmp24 = tl.full([1], 120, tl.int64)
    tmp25 = tmp21 < tmp24
    tmp26 = tmp25 & tmp20
    tmp27 = tl.load(in_ptr3 + (x0 + 16*((-1024) + x1) + 18304*x2), tmp26 & xmask, other=0.0)
    tmp28 = tmp21 >= tmp24
    tmp29 = tl.full([1], 144, tl.int64)
    tmp30 = tmp21 < tmp29
    tmp31 = tmp28 & tmp20
    tmp32 = tl.load(in_ptr1 + (16384 + x0 + 16*((-120) + ((-1024) + x1)) + 16768*x2), tmp31 & xmask, other=0.0)
    tmp33 = tl.where(tmp25, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp20, tmp33, tmp34)
    tmp36 = tmp15 >= tmp18
    tmp37 = tl.full([1], 168, tl.int64)
    tmp38 = tmp15 < tmp37
    tmp39 = tmp36 & tmp12
    tmp40 = tl.load(in_ptr2 + (16384 + x0 + 16*((-144) + ((-1024) + x1)) + 16768*x2), tmp39 & xmask, other=0.0)
    tmp41 = tl.where(tmp19, tmp35, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tl.full([1], 1, tl.int32)
    tmp52 = tmp51 / tmp50
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = tmp46 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.full([1], 0, tl.int32)
    tmp61 = triton_helpers.maximum(tmp60, tmp59)
    tl.store(out_ptr0 + (x3), tmp44, xmask)
    tl.store(out_ptr1 + (x3), tmp61, xmask)
