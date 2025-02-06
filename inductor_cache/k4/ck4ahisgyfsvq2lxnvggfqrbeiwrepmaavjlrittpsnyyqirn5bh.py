
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_pow_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_pow_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (16 + 4*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (17 + 4*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (18 + 4*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (19 + 4*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (32 + 4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (33 + 4*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (34 + 4*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (35 + 4*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (48 + 4*x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (49 + 4*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (50 + 4*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (51 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15 / tmp7
    tmp17 = tmp8 + tmp16
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24 / tmp7
    tmp26 = tmp17 + tmp25
    tmp29 = tmp27 + tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33 / tmp7
    tmp35 = tmp26 + tmp34
    tmp36 = tmp35 / tmp7
    tmp37 = tmp0 * tmp0
    tmp38 = tmp1 * tmp1
    tmp39 = tmp37 + tmp38
    tmp40 = tmp3 * tmp3
    tmp41 = tmp39 + tmp40
    tmp42 = tmp5 * tmp5
    tmp43 = tmp41 + tmp42
    tmp44 = tmp43 / tmp7
    tmp45 = tmp9 * tmp9
    tmp46 = tmp10 * tmp10
    tmp47 = tmp45 + tmp46
    tmp48 = tmp12 * tmp12
    tmp49 = tmp47 + tmp48
    tmp50 = tmp14 * tmp14
    tmp51 = tmp49 + tmp50
    tmp52 = tmp51 / tmp7
    tmp53 = tmp44 + tmp52
    tmp54 = tmp18 * tmp18
    tmp55 = tmp19 * tmp19
    tmp56 = tmp54 + tmp55
    tmp57 = tmp21 * tmp21
    tmp58 = tmp56 + tmp57
    tmp59 = tmp23 * tmp23
    tmp60 = tmp58 + tmp59
    tmp61 = tmp60 / tmp7
    tmp62 = tmp53 + tmp61
    tmp63 = tmp27 * tmp27
    tmp64 = tmp28 * tmp28
    tmp65 = tmp63 + tmp64
    tmp66 = tmp30 * tmp30
    tmp67 = tmp65 + tmp66
    tmp68 = tmp32 * tmp32
    tmp69 = tmp67 + tmp68
    tmp70 = tmp69 / tmp7
    tmp71 = tmp62 + tmp70
    tmp72 = tmp71 / tmp7
    tl.store(out_ptr0 + (x0), tmp36, xmask)
    tl.store(out_ptr1 + (x0), tmp72, xmask)
