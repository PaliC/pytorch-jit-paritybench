
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eye_mul_pow_rsub_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_eye_mul_pow_rsub_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp12 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (4 + x0), xmask)
    tmp22 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (8 + x0), xmask)
    tmp32 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (12 + x0), xmask)
    tmp2 = tmp0 * tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 == tmp4
    tmp6 = 1.0
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp6 - tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tmp10 * tmp10
    tmp14 = tmp12 * tmp13
    tmp15 = tl.full([1], 1, tl.int64)
    tmp16 = tmp3 == tmp15
    tmp17 = tl.where(tmp16, tmp6, tmp7)
    tmp18 = tmp6 - tmp17
    tmp19 = tmp14 * tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tmp11 + tmp20
    tmp24 = tmp22 * tmp23
    tmp25 = tl.full([1], 2, tl.int64)
    tmp26 = tmp3 == tmp25
    tmp27 = tl.where(tmp26, tmp6, tmp7)
    tmp28 = tmp6 - tmp27
    tmp29 = tmp24 * tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tmp21 + tmp30
    tmp34 = tmp32 * tmp33
    tmp35 = tl.full([1], 3, tl.int64)
    tmp36 = tmp3 == tmp35
    tmp37 = tl.where(tmp36, tmp6, tmp7)
    tmp38 = tmp6 - tmp37
    tmp39 = tmp34 * tmp38
    tmp40 = tmp39 * tmp39
    tmp41 = tmp31 + tmp40
    tl.store(out_ptr0 + (x0), tmp41, xmask)
