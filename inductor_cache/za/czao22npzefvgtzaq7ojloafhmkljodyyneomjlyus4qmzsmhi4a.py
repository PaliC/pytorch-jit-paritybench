
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_leaky_relu_mul_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_leaky_relu_mul_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (64 + 4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (65 + 4*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (66 + 4*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (67 + 4*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (128 + 4*x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (129 + 4*x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (130 + 4*x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (131 + 4*x0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (192 + 4*x0), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (193 + 4*x0), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr0 + (194 + 4*x0), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (195 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = 4.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp1
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp1
    tmp11 = tmp8 + tmp10
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.2
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp18 = tmp17 * tmp1
    tmp20 = tmp19 * tmp1
    tmp21 = tmp18 + tmp20
    tmp23 = tmp22 * tmp1
    tmp24 = tmp21 + tmp23
    tmp26 = tmp25 * tmp1
    tmp27 = tmp24 + tmp26
    tmp28 = tmp27 > tmp12
    tmp29 = tmp27 * tmp14
    tmp30 = tl.where(tmp28, tmp27, tmp29)
    tmp31 = triton_helpers.maximum(tmp16, tmp30)
    tmp33 = tmp32 * tmp1
    tmp35 = tmp34 * tmp1
    tmp36 = tmp33 + tmp35
    tmp38 = tmp37 * tmp1
    tmp39 = tmp36 + tmp38
    tmp41 = tmp40 * tmp1
    tmp42 = tmp39 + tmp41
    tmp43 = tmp42 > tmp12
    tmp44 = tmp42 * tmp14
    tmp45 = tl.where(tmp43, tmp42, tmp44)
    tmp46 = triton_helpers.maximum(tmp31, tmp45)
    tmp48 = tmp47 * tmp1
    tmp50 = tmp49 * tmp1
    tmp51 = tmp48 + tmp50
    tmp53 = tmp52 * tmp1
    tmp54 = tmp51 + tmp53
    tmp56 = tmp55 * tmp1
    tmp57 = tmp54 + tmp56
    tmp58 = tmp57 > tmp12
    tmp59 = tmp57 * tmp14
    tmp60 = tl.where(tmp58, tmp57, tmp59)
    tmp61 = triton_helpers.maximum(tmp46, tmp60)
    tl.store(out_ptr0 + (x0), tmp61, xmask)
