
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_max_mul_rsub_sub_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_exp_max_mul_rsub_sub_sum_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 16*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (4 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (2 + 16*x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (8 + x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (3 + 16*x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (12 + x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (4 + x0 + 16*x1), xmask)
    tmp40 = tl.load(in_ptr0 + (x0 + 16*x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp5 * tmp1
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp4, tmp8)
    tmp11 = tmp10 * tmp1
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(tmp9, tmp13)
    tmp16 = tmp15 * tmp1
    tmp18 = tmp16 + tmp17
    tmp19 = triton_helpers.maximum(tmp14, tmp18)
    tmp20 = tmp4 - tmp19
    tmp21 = tl_math.exp(tmp20)
    tmp22 = tmp8 - tmp19
    tmp23 = tl_math.exp(tmp22)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp13 - tmp19
    tmp26 = tl_math.exp(tmp25)
    tmp27 = tmp24 + tmp26
    tmp28 = tmp18 - tmp19
    tmp29 = tl_math.exp(tmp28)
    tmp30 = tmp27 + tmp29
    tmp32 = tl_math.log(tmp30)
    tmp33 = tmp19 + tmp32
    tmp35 = tmp34 * tmp31
    tmp36 = tmp33 + tmp35
    tmp37 = tmp31 * tmp36
    tmp38 = 1.0
    tmp39 = tmp38 - tmp31
    tmp41 = tmp40 * tmp1
    tmp42 = tmp39 * tmp41
    tmp43 = tmp37 + tmp42
    tl.store(in_out_ptr0 + (x2), tmp43, xmask)
