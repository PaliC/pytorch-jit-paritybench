
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_pow_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_pow_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 13456)
    x1 = xindex // 13456
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = 0.01
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp4 + tmp8
    tmp11 = tmp0 * tmp3
    tmp12 = tmp10 - tmp11
    tmp13 = 1.0208333333333333
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14 * tmp1
    tmp16 = 0.03
    tmp17 = tmp5 * tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp0 * tmp0
    tmp22 = tmp3 * tmp3
    tmp23 = tmp21 + tmp22
    tmp24 = tmp23 + tmp8
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 * tmp13
    tmp29 = tmp28 - tmp22
    tmp30 = tmp29 * tmp13
    tmp31 = tmp27 + tmp30
    tmp32 = tmp31 + tmp18
    tmp33 = tmp24 * tmp32
    tmp34 = tmp20 / tmp33
    tl.store(out_ptr0 + (x2), tmp34, xmask)
