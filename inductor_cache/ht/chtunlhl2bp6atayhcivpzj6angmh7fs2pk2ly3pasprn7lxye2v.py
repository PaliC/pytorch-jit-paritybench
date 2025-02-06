
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_clamp_convolution_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_clamp_convolution_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -10.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 10.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp8 = tmp7 + tmp1
    tmp9 = triton_helpers.maximum(tmp8, tmp3)
    tmp10 = triton_helpers.minimum(tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp6, tmp10)
    tmp13 = tmp12 + tmp1
    tmp14 = triton_helpers.maximum(tmp13, tmp3)
    tmp15 = triton_helpers.minimum(tmp14, tmp5)
    tmp16 = triton_helpers.maximum(tmp11, tmp15)
    tmp18 = tmp17 + tmp1
    tmp19 = triton_helpers.maximum(tmp18, tmp3)
    tmp20 = triton_helpers.minimum(tmp19, tmp5)
    tmp21 = triton_helpers.maximum(tmp16, tmp20)
    tmp22 = tmp6 - tmp21
    tmp23 = tl_math.exp(tmp22)
    tmp24 = tmp10 - tmp21
    tmp25 = tl_math.exp(tmp24)
    tmp26 = tmp23 + tmp25
    tmp27 = tmp15 - tmp21
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tmp26 + tmp28
    tmp30 = tmp20 - tmp21
    tmp31 = tl_math.exp(tmp30)
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr0 + (x2), tmp21, xmask)
    tl.store(out_ptr1 + (x2), tmp32, xmask)
