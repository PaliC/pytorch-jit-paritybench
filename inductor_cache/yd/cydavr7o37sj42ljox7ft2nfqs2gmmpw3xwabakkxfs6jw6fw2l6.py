
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_log_sigmoid_forward_mean_mul_neg_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_log_sigmoid_forward_mean_mul_neg_0(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp10 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.minimum(tmp3, tmp2)
    tmp5 = tl_math.abs(tmp2)
    tmp6 = -tmp5
    tmp7 = tl_math.exp(tmp6)
    tmp8 = libdevice.log1p(tmp7)
    tmp9 = tmp4 - tmp8
    tmp11 = tmp10 * tmp1
    tmp12 = -tmp11
    tmp13 = triton_helpers.minimum(tmp3, tmp12)
    tmp14 = tl_math.abs(tmp12)
    tmp15 = -tmp14
    tmp16 = tl_math.exp(tmp15)
    tmp17 = libdevice.log1p(tmp16)
    tmp18 = tmp13 - tmp17
    tmp20 = tmp19 * tmp1
    tmp21 = -tmp20
    tmp22 = triton_helpers.minimum(tmp3, tmp21)
    tmp23 = tl_math.abs(tmp21)
    tmp24 = -tmp23
    tmp25 = tl_math.exp(tmp24)
    tmp26 = libdevice.log1p(tmp25)
    tmp27 = tmp22 - tmp26
    tmp28 = tmp18 + tmp27
    tmp30 = tmp29 * tmp1
    tmp31 = -tmp30
    tmp32 = triton_helpers.minimum(tmp3, tmp31)
    tmp33 = tl_math.abs(tmp31)
    tmp34 = -tmp33
    tmp35 = tl_math.exp(tmp34)
    tmp36 = libdevice.log1p(tmp35)
    tmp37 = tmp32 - tmp36
    tmp38 = tmp28 + tmp37
    tmp40 = tmp39 * tmp1
    tmp41 = -tmp40
    tmp42 = triton_helpers.minimum(tmp3, tmp41)
    tmp43 = tl_math.abs(tmp41)
    tmp44 = -tmp43
    tmp45 = tl_math.exp(tmp44)
    tmp46 = libdevice.log1p(tmp45)
    tmp47 = tmp42 - tmp46
    tmp48 = tmp38 + tmp47
    tmp49 = 4.0
    tmp50 = tmp48 / tmp49
    tmp51 = tmp50 * tmp1
    tmp52 = tmp9 + tmp51
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
    tl.store(out_ptr0 + (x0), tmp50, xmask)
    tl.store(out_ptr1 + (x0), tmp52, xmask)
