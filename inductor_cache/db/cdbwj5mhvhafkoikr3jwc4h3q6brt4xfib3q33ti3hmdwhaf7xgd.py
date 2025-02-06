
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_div_le_mul_pow_rsub_sqrt_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_div_le_mul_pow_rsub_sqrt_sub_sum_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr2 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr2 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp3 / tmp1
    tmp5 = tmp2 * tmp2
    tmp6 = tmp4 - tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp1 - tmp7
    tmp9 = tmp8 / tmp1
    tmp10 = 0.0
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tmp2 - tmp12
    tmp15 = tmp13 <= tmp14
    tmp16 = tmp15.to(tl.int64)
    tmp18 = 2.0
    tmp19 = tmp17 / tmp18
    tmp21 = tmp20 / tmp18
    tmp22 = tmp19 * tmp19
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = tmp1 - tmp24
    tmp26 = tmp25 / tmp18
    tmp27 = triton_helpers.maximum(tmp26, tmp10)
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tmp19 - tmp28
    tmp31 = tmp29 <= tmp30
    tmp32 = tmp31.to(tl.int64)
    tmp33 = tmp16 + tmp32
    tmp35 = 3.0
    tmp36 = tmp34 / tmp35
    tmp38 = tmp37 / tmp35
    tmp39 = tmp36 * tmp36
    tmp40 = tmp38 - tmp39
    tmp41 = tmp35 * tmp40
    tmp42 = tmp1 - tmp41
    tmp43 = tmp42 / tmp35
    tmp44 = triton_helpers.maximum(tmp43, tmp10)
    tmp45 = libdevice.sqrt(tmp44)
    tmp46 = tmp36 - tmp45
    tmp48 = tmp46 <= tmp47
    tmp49 = tmp48.to(tl.int64)
    tmp50 = tmp33 + tmp49
    tmp52 = 4.0
    tmp53 = tmp51 / tmp52
    tmp55 = tmp54 / tmp52
    tmp56 = tmp53 * tmp53
    tmp57 = tmp55 - tmp56
    tmp58 = tmp52 * tmp57
    tmp59 = tmp1 - tmp58
    tmp60 = tmp59 / tmp52
    tmp61 = triton_helpers.maximum(tmp60, tmp10)
    tmp62 = libdevice.sqrt(tmp61)
    tmp63 = tmp53 - tmp62
    tmp65 = tmp63 <= tmp64
    tmp66 = tmp65.to(tl.int64)
    tmp67 = tmp50 + tmp66
    tl.store(out_ptr0 + (x0), tmp67, xmask)
