
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_exp_log_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 21, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_exp_log_mul_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (7*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (7*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), xmask)
    tmp15 = tl.load(in_ptr1 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask)
    tmp25 = tl.load(in_ptr1 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr2 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr5 + (x2), xmask)
    tmp35 = tl.load(in_ptr1 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr2 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr6 + (x2), xmask)
    tmp45 = tl.load(in_ptr1 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr2 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr7 + (x2), xmask)
    tmp55 = tl.load(in_ptr1 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr2 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr8 + (x2), xmask)
    tmp65 = tl.load(in_ptr1 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr2 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp1 = 10.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 * tmp5
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tmp3 * tmp10
    tmp13 = tmp12 * tmp1
    tmp14 = tl_math.exp(tmp13)
    tmp16 = tmp15 * tmp5
    tmp18 = tmp17 * tmp8
    tmp19 = tmp16 + tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp11 + tmp20
    tmp23 = tmp22 * tmp1
    tmp24 = tl_math.exp(tmp23)
    tmp26 = tmp25 * tmp5
    tmp28 = tmp27 * tmp8
    tmp29 = tmp26 + tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = tmp21 + tmp30
    tmp33 = tmp32 * tmp1
    tmp34 = tl_math.exp(tmp33)
    tmp36 = tmp35 * tmp5
    tmp38 = tmp37 * tmp8
    tmp39 = tmp36 + tmp38
    tmp40 = tmp34 * tmp39
    tmp41 = tmp31 + tmp40
    tmp43 = tmp42 * tmp1
    tmp44 = tl_math.exp(tmp43)
    tmp46 = tmp45 * tmp5
    tmp48 = tmp47 * tmp8
    tmp49 = tmp46 + tmp48
    tmp50 = tmp44 * tmp49
    tmp51 = tmp41 + tmp50
    tmp53 = tmp52 * tmp1
    tmp54 = tl_math.exp(tmp53)
    tmp56 = tmp55 * tmp5
    tmp58 = tmp57 * tmp8
    tmp59 = tmp56 + tmp58
    tmp60 = tmp54 * tmp59
    tmp61 = tmp51 + tmp60
    tmp63 = tmp62 * tmp1
    tmp64 = tl_math.exp(tmp63)
    tmp66 = tmp65 * tmp5
    tmp68 = tmp67 * tmp8
    tmp69 = tmp66 + tmp68
    tmp70 = tmp64 * tmp69
    tmp71 = tmp61 + tmp70
    tmp72 = tl_math.log(tmp71)
    tl.store(in_out_ptr0 + (x2), tmp71, xmask)
    tl.store(out_ptr0 + (x2), tmp72, xmask)
