
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_relu_threshold_backward_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_relu_threshold_backward_12(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2)
    x1 = xindex // 2
    x5 = xindex
    x6 = xindex // 4
    x3 = ((xindex // 4) % 512)
    tmp0 = tl.load(in_ptr0 + (4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (8 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (9 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (10 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (11 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (16 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (17 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (18 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (19 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (24 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (25 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (26 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (27 + 4*x0 + 32*x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr1 + (x6), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp31 = 0.0625
    tmp32 = tmp30 * tmp31
    tmp33 = tl.full([1], 0, tl.int32)
    tmp34 = triton_helpers.maximum(tmp33, tmp32)
    tmp37 = tmp35 + tmp36
    tmp38 = tmp34 + tmp37
    tmp39 = 0.0
    tmp40 = tmp34 <= tmp39
    tl.store(out_ptr1 + (x5), tmp38, None)
    tl.store(out_ptr2 + (x5), tmp40, None)
