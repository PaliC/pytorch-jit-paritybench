
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_mean_mul_rsqrt_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_leaky_relu_mean_mul_rsqrt_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 49)
    x2 = xindex // 196
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp6 = tl.load(in_ptr0 + (x0 + 196*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (49 + x0 + 196*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (98 + x0 + 196*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (147 + x0 + 196*x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp6 > tmp1
    tmp8 = tmp6 * tmp3
    tmp9 = tl.where(tmp7, tmp6, tmp8)
    tmp10 = tmp9 * tmp9
    tmp12 = tmp11 > tmp1
    tmp13 = tmp11 * tmp3
    tmp14 = tl.where(tmp12, tmp11, tmp13)
    tmp15 = tmp14 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp17 > tmp1
    tmp19 = tmp17 * tmp3
    tmp20 = tl.where(tmp18, tmp17, tmp19)
    tmp21 = tmp20 * tmp20
    tmp22 = tmp16 + tmp21
    tmp24 = tmp23 > tmp1
    tmp25 = tmp23 * tmp3
    tmp26 = tl.where(tmp24, tmp23, tmp25)
    tmp27 = tmp26 * tmp26
    tmp28 = tmp22 + tmp27
    tmp29 = 4.0
    tmp30 = tmp28 / tmp29
    tmp31 = 1e-08
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp5 * tmp33
    tl.store(out_ptr0 + (x3), tmp34, xmask)
