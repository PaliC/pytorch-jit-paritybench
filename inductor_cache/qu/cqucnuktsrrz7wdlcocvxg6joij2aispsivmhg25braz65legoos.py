
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 19, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_mul_4(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (7*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (7*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr1 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr2 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr2 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr2 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr1 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr2 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr1 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr2 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-12
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 / tmp14
    tmp17 = 0.0
    tmp18 = tmp16 * tmp17
    tmp20 = 1.0
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tmp15 * tmp22
    tmp25 = tmp24 * tmp17
    tmp27 = tmp26 * tmp20
    tmp28 = tmp25 + tmp27
    tmp29 = tmp15 * tmp28
    tmp31 = tmp30 * tmp17
    tmp33 = tmp32 * tmp20
    tmp34 = tmp31 + tmp33
    tmp35 = tmp15 * tmp34
    tmp37 = tmp36 * tmp17
    tmp39 = tmp38 * tmp20
    tmp40 = tmp37 + tmp39
    tmp41 = tmp15 * tmp40
    tmp43 = tmp42 * tmp17
    tmp45 = tmp44 * tmp20
    tmp46 = tmp43 + tmp45
    tmp47 = tmp15 * tmp46
    tmp49 = tmp48 * tmp17
    tmp51 = tmp50 * tmp20
    tmp52 = tmp49 + tmp51
    tmp53 = tmp15 * tmp52
    tmp55 = tmp54 * tmp17
    tmp57 = tmp56 * tmp20
    tmp58 = tmp55 + tmp57
    tmp59 = tmp15 * tmp58
    tl.store(out_ptr1 + (x2), tmp23, xmask)
    tl.store(out_ptr2 + (x2), tmp29, xmask)
    tl.store(out_ptr3 + (x2), tmp35, xmask)
    tl.store(out_ptr4 + (x2), tmp41, xmask)
    tl.store(out_ptr5 + (x2), tmp47, xmask)
    tl.store(out_ptr6 + (x2), tmp53, xmask)
    tl.store(out_ptr7 + (x2), tmp59, xmask)
