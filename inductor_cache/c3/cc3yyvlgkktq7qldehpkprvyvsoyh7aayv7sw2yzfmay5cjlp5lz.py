
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
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_exp_log_mul_pow_sub_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_exp_log_mul_pow_sub_sum_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (2))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp19 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (3))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp28 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = libdevice.exp(tmp4)
    tmp6 = tmp3 / tmp5
    tmp7 = tmp6 * tmp6
    tmp11 = tmp9 - tmp10
    tmp13 = libdevice.exp(tmp12)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tmp7 + tmp15
    tmp20 = tmp18 - tmp19
    tmp22 = libdevice.exp(tmp21)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tmp16 + tmp24
    tmp29 = tmp27 - tmp28
    tmp31 = libdevice.exp(tmp30)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tmp25 + tmp33
    tmp36 = libdevice.log(tmp35)
    tmp37 = tl.full([1], -3.6757541328186907, tl.float64)
    tmp38 = tmp37 + tmp36
    tmp39 = tl.full([1], 0.5, tl.float64)
    tmp40 = tmp34 * tmp39
    tmp41 = tmp38 - tmp40
    tmp42 = tmp4 + tmp12
    tmp43 = tmp42 + tmp21
    tmp44 = tmp43 + tmp30
    tmp45 = tmp41 - tmp44
    tl.store(in_out_ptr0 + (x0), tmp45, xmask)
