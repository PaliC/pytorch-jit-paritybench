
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_binary_cross_entropy_with_logits_mean_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_binary_cross_entropy_with_logits_mean_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 96
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (4*x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr1 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr1 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.minimum(tmp5, tmp3)
    tmp7 = tl_math.abs(tmp3)
    tmp8 = -tmp7
    tmp9 = tl_math.exp(tmp8)
    tmp10 = libdevice.log1p(tmp9)
    tmp11 = tmp6 - tmp10
    tmp12 = tmp4 - tmp11
    tmp14 = tmp1 - tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = triton_helpers.minimum(tmp5, tmp15)
    tmp18 = tl_math.abs(tmp15)
    tmp19 = -tmp18
    tmp20 = tl_math.exp(tmp19)
    tmp21 = libdevice.log1p(tmp20)
    tmp22 = tmp17 - tmp21
    tmp23 = tmp16 - tmp22
    tmp24 = tmp12 + tmp23
    tmp26 = tmp1 - tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = triton_helpers.minimum(tmp5, tmp27)
    tmp30 = tl_math.abs(tmp27)
    tmp31 = -tmp30
    tmp32 = tl_math.exp(tmp31)
    tmp33 = libdevice.log1p(tmp32)
    tmp34 = tmp29 - tmp33
    tmp35 = tmp28 - tmp34
    tmp36 = tmp24 + tmp35
    tmp38 = tmp1 - tmp37
    tmp40 = tmp38 * tmp39
    tmp41 = triton_helpers.minimum(tmp5, tmp39)
    tmp42 = tl_math.abs(tmp39)
    tmp43 = -tmp42
    tmp44 = tl_math.exp(tmp43)
    tmp45 = libdevice.log1p(tmp44)
    tmp46 = tmp41 - tmp45
    tmp47 = tmp40 - tmp46
    tmp48 = tmp36 + tmp47
    tmp49 = 4.0
    tmp50 = tmp48 / tmp49
    tl.store(out_ptr0 + (x2), tmp50, xmask)
