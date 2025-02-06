
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_maximum_mean_minimum_mul_pow_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_maximum_mean_minimum_mul_pow_sub_sum_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp2 - tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = 0.0
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = tmp13 - tmp16
    tmp18 = tmp17 + tmp7
    tmp19 = triton_helpers.maximum(tmp18, tmp9)
    tmp20 = tmp10 * tmp19
    tmp21 = tmp0 - tmp3
    tmp22 = tmp11 - tmp14
    tmp23 = tmp21 * tmp22
    tmp24 = tmp1 - tmp4
    tmp25 = tmp12 - tmp15
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp20
    tmp29 = tmp3 + tmp0
    tmp30 = 2.0
    tmp31 = tmp29 / tmp30
    tmp32 = tmp4 + tmp1
    tmp33 = tmp32 / tmp30
    tmp34 = tmp31 - tmp33
    tmp35 = tmp34 * tmp34
    tmp36 = tmp14 + tmp11
    tmp37 = tmp36 / tmp30
    tmp38 = tmp15 + tmp12
    tmp39 = tmp38 / tmp30
    tmp40 = tmp37 - tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = tmp35 + tmp41
    tmp43 = triton_helpers.minimum(tmp3, tmp4)
    tmp44 = triton_helpers.maximum(tmp0, tmp1)
    tmp45 = tmp43 - tmp44
    tmp46 = tmp45 * tmp45
    tmp47 = triton_helpers.minimum(tmp14, tmp15)
    tmp48 = triton_helpers.maximum(tmp11, tmp12)
    tmp49 = tmp47 - tmp48
    tmp50 = tmp49 * tmp49
    tmp51 = tmp46 + tmp50
    tmp52 = tmp51 + tmp7
    tmp53 = tmp42 / tmp52
    tl.store(out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr1 + (x0), tmp28, xmask)
    tl.store(out_ptr2 + (x0), tmp53, xmask)
