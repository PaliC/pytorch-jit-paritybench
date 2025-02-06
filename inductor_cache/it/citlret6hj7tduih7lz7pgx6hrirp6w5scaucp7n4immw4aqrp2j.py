
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_log_neg_pow_sub_sum_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_log_neg_pow_sub_sum_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (1))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp31 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr2 + (2))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp46 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr2 + (3))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp1 = tmp0 * tmp0
    tmp2 = -tmp1
    tmp4 = tmp2 / tmp3
    tmp7 = 20.0
    tmp8 = tmp6 > tmp7
    tmp9 = tl_math.exp(tmp6)
    tmp10 = libdevice.log1p(tmp9)
    tmp11 = tl.where(tmp8, tmp6, tmp10)
    tmp12 = tl_math.log(tmp11)
    tmp13 = tmp4 - tmp12
    tmp14 = 0.9189385332046727
    tmp15 = tmp13 - tmp14
    tmp17 = tmp16 * tmp16
    tmp18 = -tmp17
    tmp20 = tmp18 / tmp19
    tmp23 = tmp22 > tmp7
    tmp24 = tl_math.exp(tmp22)
    tmp25 = libdevice.log1p(tmp24)
    tmp26 = tl.where(tmp23, tmp22, tmp25)
    tmp27 = tl_math.log(tmp26)
    tmp28 = tmp20 - tmp27
    tmp29 = tmp28 - tmp14
    tmp30 = tmp15 + tmp29
    tmp32 = tmp31 * tmp31
    tmp33 = -tmp32
    tmp35 = tmp33 / tmp34
    tmp38 = tmp37 > tmp7
    tmp39 = tl_math.exp(tmp37)
    tmp40 = libdevice.log1p(tmp39)
    tmp41 = tl.where(tmp38, tmp37, tmp40)
    tmp42 = tl_math.log(tmp41)
    tmp43 = tmp35 - tmp42
    tmp44 = tmp43 - tmp14
    tmp45 = tmp30 + tmp44
    tmp47 = tmp46 * tmp46
    tmp48 = -tmp47
    tmp50 = tmp48 / tmp49
    tmp53 = tmp52 > tmp7
    tmp54 = tl_math.exp(tmp52)
    tmp55 = libdevice.log1p(tmp54)
    tmp56 = tl.where(tmp53, tmp52, tmp55)
    tmp57 = tl_math.log(tmp56)
    tmp58 = tmp50 - tmp57
    tmp59 = tmp58 - tmp14
    tmp60 = tmp45 + tmp59
    tmp61 = 1.4189385332046727
    tmp62 = tmp12 + tmp61
    tmp63 = tmp27 + tmp61
    tmp64 = tmp62 + tmp63
    tmp65 = tmp42 + tmp61
    tmp66 = tmp64 + tmp65
    tmp67 = tmp57 + tmp61
    tmp68 = tmp66 + tmp67
    tl.store(out_ptr0 + (x0), tmp60, xmask)
    tl.store(out_ptr1 + (x0), tmp68, xmask)
