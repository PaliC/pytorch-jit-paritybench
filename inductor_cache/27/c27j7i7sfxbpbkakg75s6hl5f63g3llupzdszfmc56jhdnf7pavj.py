
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_pow_sqrt_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 32, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_pow_sqrt_sub_sum_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (16*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr1 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr1 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr1 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr1 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr1 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr1 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr1 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr1 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp76 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr1 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp6 = tmp4 - tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tmp3 + tmp7
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tmp8 + tmp12
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp13 + tmp17
    tmp19 = libdevice.sqrt(tmp18)
    tmp22 = tmp20 - tmp21
    tmp23 = tmp22 * tmp22
    tmp26 = tmp24 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tmp23 + tmp27
    tmp31 = tmp29 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tmp28 + tmp32
    tmp36 = tmp34 - tmp35
    tmp37 = tmp36 * tmp36
    tmp38 = tmp33 + tmp37
    tmp39 = libdevice.sqrt(tmp38)
    tmp40 = tmp19 + tmp39
    tmp43 = tmp41 - tmp42
    tmp44 = tmp43 * tmp43
    tmp47 = tmp45 - tmp46
    tmp48 = tmp47 * tmp47
    tmp49 = tmp44 + tmp48
    tmp52 = tmp50 - tmp51
    tmp53 = tmp52 * tmp52
    tmp54 = tmp49 + tmp53
    tmp57 = tmp55 - tmp56
    tmp58 = tmp57 * tmp57
    tmp59 = tmp54 + tmp58
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp40 + tmp60
    tmp64 = tmp62 - tmp63
    tmp65 = tmp64 * tmp64
    tmp68 = tmp66 - tmp67
    tmp69 = tmp68 * tmp68
    tmp70 = tmp65 + tmp69
    tmp73 = tmp71 - tmp72
    tmp74 = tmp73 * tmp73
    tmp75 = tmp70 + tmp74
    tmp78 = tmp76 - tmp77
    tmp79 = tmp78 * tmp78
    tmp80 = tmp75 + tmp79
    tmp81 = libdevice.sqrt(tmp80)
    tmp82 = tmp61 + tmp81
    tl.store(out_ptr0 + (x0), tmp82, xmask)
