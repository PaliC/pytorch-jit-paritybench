
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_exp_logsumexp_max_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_exp_logsumexp_max_mul_sub_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp7 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl_math.exp(tmp2)
    tmp4 = float("inf")
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp0 * tmp5
    tmp8 = tmp7 * tmp5
    tmp9 = triton_helpers.maximum(tmp6, tmp8)
    tmp11 = tmp10 * tmp5
    tmp12 = triton_helpers.maximum(tmp9, tmp11)
    tmp14 = tmp13 * tmp5
    tmp15 = triton_helpers.maximum(tmp12, tmp14)
    tmp16 = tmp6 - tmp15
    tmp17 = tmp8 - tmp15
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = tmp11 - tmp15
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp21 = tmp14 - tmp15
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = tl_math.abs(tmp22)
    tmp24 = tmp23 == tmp4
    tmp25 = 0.0
    tmp26 = tl.where(tmp24, tmp25, tmp22)
    tmp27 = tmp16 - tmp26
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tmp17 - tmp26
    tmp30 = tl_math.exp(tmp29)
    tmp31 = tmp28 + tmp30
    tmp32 = tmp19 - tmp26
    tmp33 = tl_math.exp(tmp32)
    tmp34 = tmp31 + tmp33
    tmp35 = tmp21 - tmp26
    tmp36 = tl_math.exp(tmp35)
    tmp37 = tmp34 + tmp36
    tl.store(out_ptr0 + (x0), tmp15, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
    tl.store(out_ptr2 + (x0), tmp37, xmask)
