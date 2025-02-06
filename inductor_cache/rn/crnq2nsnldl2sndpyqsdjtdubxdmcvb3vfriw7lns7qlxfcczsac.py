
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_clamp_pow_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_clamp_pow_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp1 = 1e-07
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3 * tmp2
    tmp6 = triton_helpers.maximum(tmp5, tmp1)
    tmp7 = tmp6 * tmp6
    tmp8 = tmp7 * tmp6
    tmp9 = tmp8 + tmp4
    tmp11 = triton_helpers.maximum(tmp10, tmp1)
    tmp12 = tmp11 * tmp11
    tmp13 = tmp12 * tmp11
    tmp14 = tmp13 + tmp9
    tmp16 = triton_helpers.maximum(tmp15, tmp1)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp17 * tmp16
    tmp19 = tmp18 + tmp14
    tmp21 = triton_helpers.maximum(tmp20, tmp1)
    tmp22 = tmp21 * tmp21
    tmp23 = tmp22 * tmp21
    tmp24 = tmp23 + tmp19
    tmp26 = triton_helpers.maximum(tmp25, tmp1)
    tmp27 = tmp26 * tmp26
    tmp28 = tmp27 * tmp26
    tmp29 = tmp28 + tmp24
    tmp31 = triton_helpers.maximum(tmp30, tmp1)
    tmp32 = tmp31 * tmp31
    tmp33 = tmp32 * tmp31
    tmp34 = tmp33 + tmp29
    tmp36 = triton_helpers.maximum(tmp35, tmp1)
    tmp37 = tmp36 * tmp36
    tmp38 = tmp37 * tmp36
    tmp39 = tmp38 + tmp34
    tmp41 = triton_helpers.maximum(tmp40, tmp1)
    tmp42 = tmp41 * tmp41
    tmp43 = tmp42 * tmp41
    tmp44 = tmp43 + tmp39
    tmp46 = triton_helpers.maximum(tmp45, tmp1)
    tmp47 = tmp46 * tmp46
    tmp48 = tmp47 * tmp46
    tmp49 = tmp48 + tmp44
    tmp51 = triton_helpers.maximum(tmp50, tmp1)
    tmp52 = tmp51 * tmp51
    tmp53 = tmp52 * tmp51
    tmp54 = tmp53 + tmp49
    tmp56 = triton_helpers.maximum(tmp55, tmp1)
    tmp57 = tmp56 * tmp56
    tmp58 = tmp57 * tmp56
    tmp59 = tmp58 + tmp54
    tmp61 = triton_helpers.maximum(tmp60, tmp1)
    tmp62 = tmp61 * tmp61
    tmp63 = tmp62 * tmp61
    tmp64 = tmp63 + tmp59
    tmp66 = triton_helpers.maximum(tmp65, tmp1)
    tmp67 = tmp66 * tmp66
    tmp68 = tmp67 * tmp66
    tmp69 = tmp68 + tmp64
    tmp71 = triton_helpers.maximum(tmp70, tmp1)
    tmp72 = tmp71 * tmp71
    tmp73 = tmp72 * tmp71
    tmp74 = tmp73 + tmp69
    tmp76 = triton_helpers.maximum(tmp75, tmp1)
    tmp77 = tmp76 * tmp76
    tmp78 = tmp77 * tmp76
    tmp79 = tmp78 + tmp74
    tmp80 = 0.0625
    tmp81 = tmp79 * tmp80
    tmp82 = 0.3333333333333333
    tmp83 = libdevice.pow(tmp81, tmp82)
    tl.store(in_out_ptr0 + (x0), tmp83, xmask)
