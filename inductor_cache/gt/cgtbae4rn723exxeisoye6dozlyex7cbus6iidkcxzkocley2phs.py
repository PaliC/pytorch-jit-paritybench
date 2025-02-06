
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_unfold_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_unfold_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x3 = ((xindex // 16) % 4)
    x4 = xindex // 64
    x5 = (xindex % 16)
    x2 = ((xindex // 4) % 4)
    x1 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp20 = tl.load(in_ptr3 + (x0), xmask)
    tmp33 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = x3
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr0 + (x5 + 16*(x3) + 64*x4), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x2 + 4*(x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp1 >= tmp4
    tmp12 = tl.full([1], 4, tl.int64)
    tmp13 = tmp1 < tmp12
    tmp14 = tl.load(in_ptr0 + (32 + x5 + 16*((-2) + x3) + 64*x4), tmp11 & xmask, other=0.0)
    tmp15 = tl.load(in_ptr2 + (x1 + 4*((-2) + x3)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp11, tmp16, tmp17)
    tmp19 = tl.where(tmp5, tmp10, tmp18)
    tmp21 = 0.0
    tmp22 = tmp19 >= tmp21
    tmp23 = 1.0
    tmp24 = -1.0
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tmp20 * tmp25
    tmp27 = tmp26 - tmp26
    tmp28 = tmp25 * tmp19
    tmp29 = tmp27 * tmp28
    tmp30 = tl_math.exp(tmp29)
    tmp31 = tmp30 / tmp30
    tmp32 = tmp31 * tmp0
    tmp34 = tmp32 - tmp33
    tmp36 = 1e-05
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.sqrt(tmp37)
    tmp39 = tl.full([1], 1, tl.int32)
    tmp40 = tmp39 / tmp38
    tmp41 = tmp40 * tmp23
    tmp42 = tmp34 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tl.full([1], 0, tl.int32)
    tmp48 = triton_helpers.maximum(tmp47, tmp46)
    tl.store(in_out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr0 + (x0), tmp19, xmask)
    tl.store(out_ptr1 + (x0), tmp48, xmask)
