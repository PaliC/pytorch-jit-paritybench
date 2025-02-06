
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_max_mul_neg_pow_reciprocal_relu_sub_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_max_mul_neg_pow_reciprocal_relu_sub_sum_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 * tmp1
    tmp3 = 1.0
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = -0.3333333333333333
    tmp8 = libdevice.pow(tmp6, tmp7)
    tmp10 = tmp9 * tmp1
    tmp11 = tmp10 + tmp3
    tmp12 = triton_helpers.maximum(tmp5, tmp11)
    tmp13 = libdevice.pow(tmp12, tmp7)
    tmp14 = tmp8 + tmp13
    tmp16 = tmp15 * tmp1
    tmp17 = tmp16 + tmp3
    tmp18 = triton_helpers.maximum(tmp5, tmp17)
    tmp19 = libdevice.pow(tmp18, tmp7)
    tmp20 = tmp14 + tmp19
    tmp22 = tmp21 * tmp1
    tmp23 = tmp22 + tmp3
    tmp24 = triton_helpers.maximum(tmp5, tmp23)
    tmp25 = libdevice.pow(tmp24, tmp7)
    tmp26 = tmp20 + tmp25
    tmp27 = tl.full([1], 1, tl.int32)
    tmp28 = tmp27 / tmp26
    tmp29 = tmp28 * tmp3
    tmp30 = tmp27 / tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tmp31 * tmp30
    tmp33 = tmp32 - tmp3
    tmp34 = tmp33 * tmp7
    tmp35 = -tmp34
    tmp38 = triton_helpers.maximum(tmp36, tmp37)
    tmp40 = triton_helpers.maximum(tmp38, tmp39)
    tmp42 = triton_helpers.maximum(tmp40, tmp41)
    tmp43 = tmp35 + tmp42
    tl.store(in_out_ptr0 + (x0), tmp43, xmask)
