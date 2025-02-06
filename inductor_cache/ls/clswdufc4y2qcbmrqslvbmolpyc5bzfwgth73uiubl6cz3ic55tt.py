
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_log_mv_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_log_mv_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr2 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (4 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (1))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp18 = tl.load(in_ptr2 + (1))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp22 = tl.load(in_ptr0 + (8 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (2))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp29 = tl.load(in_ptr2 + (2))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp33 = tl.load(in_ptr0 + (12 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr1 + (3))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK])
    tmp40 = tl.load(in_ptr2 + (3))
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
    tmp1 = libdevice.tanh(tmp0)
    tmp2 = tmp1 * tmp1
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp7 = tmp4 * tmp6
    tmp10 = tmp7 * tmp9
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = tmp12 * tmp12
    tmp14 = tmp3 - tmp13
    tmp17 = tmp14 * tmp16
    tmp20 = tmp17 * tmp19
    tmp21 = tmp10 + tmp20
    tmp23 = libdevice.tanh(tmp22)
    tmp24 = tmp23 * tmp23
    tmp25 = tmp3 - tmp24
    tmp28 = tmp25 * tmp27
    tmp31 = tmp28 * tmp30
    tmp32 = tmp21 + tmp31
    tmp34 = libdevice.tanh(tmp33)
    tmp35 = tmp34 * tmp34
    tmp36 = tmp3 - tmp35
    tmp39 = tmp36 * tmp38
    tmp42 = tmp39 * tmp41
    tmp43 = tmp32 + tmp42
    tmp44 = tmp43 + tmp3
    tmp45 = tl_math.abs(tmp44)
    tmp46 = 1e-15
    tmp47 = tmp45 + tmp46
    tmp48 = tl_math.log(tmp47)
    tl.store(in_out_ptr0 + (x0), tmp48, xmask)
