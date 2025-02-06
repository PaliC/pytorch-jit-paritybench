
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = tmp0 - tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tmp1 - tmp12
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tmp14 + tmp16
    tmp18 = tmp3 - tmp12
    tmp19 = tl_math.exp(tmp18)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp5 - tmp12
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp20 + tmp22
    tmp24 = tmp7 - tmp12
    tmp25 = tl_math.exp(tmp24)
    tmp26 = tmp23 + tmp25
    tmp27 = tmp9 - tmp12
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tmp26 + tmp28
    tmp30 = tmp11 - tmp12
    tmp31 = tl_math.exp(tmp30)
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp32, xmask)
