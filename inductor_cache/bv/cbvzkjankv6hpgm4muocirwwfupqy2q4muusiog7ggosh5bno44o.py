
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_mul_rsub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_mul_rsub_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp8 = tl.load(in_ptr2 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + (1))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp19 = tl.load(in_ptr1 + (1))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp22 = tl.load(in_ptr2 + (1))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp30 = tl.load(in_ptr0 + (2))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp33 = tl.load(in_ptr1 + (2))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp36 = tl.load(in_ptr2 + (2))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp44 = tl.load(in_ptr0 + (3))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp47 = tl.load(in_ptr1 + (3))
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK])
    tmp50 = tl.load(in_ptr2 + (3))
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK])
    tmp2 = 2.0
    tmp3 = tmp1 * tmp2
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp10 = tmp9 + tmp6
    tmp11 = tmp7 + tmp10
    tmp12 = tmp3 / tmp11
    tmp13 = 1.0
    tmp14 = tmp13 - tmp12
    tmp15 = tmp14 * tmp13
    tmp18 = tmp17 * tmp2
    tmp21 = tmp20 + tmp6
    tmp24 = tmp23 + tmp6
    tmp25 = tmp21 + tmp24
    tmp26 = tmp18 / tmp25
    tmp27 = tmp13 - tmp26
    tmp28 = tmp27 * tmp13
    tmp29 = tmp15 + tmp28
    tmp32 = tmp31 * tmp2
    tmp35 = tmp34 + tmp6
    tmp38 = tmp37 + tmp6
    tmp39 = tmp35 + tmp38
    tmp40 = tmp32 / tmp39
    tmp41 = tmp13 - tmp40
    tmp42 = tmp41 * tmp13
    tmp43 = tmp29 + tmp42
    tmp46 = tmp45 * tmp2
    tmp49 = tmp48 + tmp6
    tmp52 = tmp51 + tmp6
    tmp53 = tmp49 + tmp52
    tmp54 = tmp46 / tmp53
    tmp55 = tmp13 - tmp54
    tmp56 = tmp55 * tmp13
    tmp57 = tmp43 + tmp56
    tmp58 = 4.0
    tmp59 = tmp57 / tmp58
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp59, None)
