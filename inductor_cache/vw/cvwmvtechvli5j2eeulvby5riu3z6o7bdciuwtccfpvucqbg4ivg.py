
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_div_eq_masked_fill_mul_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_div_eq_masked_fill_mul_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (4*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = 0.0
    tmp3 = tmp1 == tmp2
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = -10000.0
    tmp8 = tl.where(tmp3, tmp7, tmp6)
    tmp10 = tmp9 * tmp5
    tmp11 = tl.where(tmp3, tmp7, tmp10)
    tmp12 = triton_helpers.maximum(tmp8, tmp11)
    tmp14 = tmp13 * tmp5
    tmp15 = tl.where(tmp3, tmp7, tmp14)
    tmp16 = triton_helpers.maximum(tmp12, tmp15)
    tmp18 = tmp17 * tmp5
    tmp19 = tl.where(tmp3, tmp7, tmp18)
    tmp20 = triton_helpers.maximum(tmp16, tmp19)
    tmp21 = tmp8 - tmp20
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp11 - tmp20
    tmp24 = tl_math.exp(tmp23)
    tmp25 = tmp22 + tmp24
    tmp26 = tmp15 - tmp20
    tmp27 = tl_math.exp(tmp26)
    tmp28 = tmp25 + tmp27
    tmp29 = tmp19 - tmp20
    tmp30 = tl_math.exp(tmp29)
    tmp31 = tmp28 + tmp30
    tl.store(out_ptr0 + (x2), tmp20, xmask)
    tl.store(out_ptr1 + (x2), tmp31, xmask)
