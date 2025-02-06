
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_sub_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_sub_sum_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr0 + (1))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (2))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (3))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp13 = tl.load(in_ptr1 + (0))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp15 = tl.load(in_ptr1 + (1))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp18 = tl.load(in_ptr1 + (2))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp21 = tl.load(in_ptr1 + (3))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + (4))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp26 = tl.load(in_ptr1 + (5))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp29 = tl.load(in_ptr1 + (6))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp32 = tl.load(in_ptr1 + (7))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp36 = tl.load(in_ptr1 + (8))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp38 = tl.load(in_ptr1 + (9))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp41 = tl.load(in_ptr1 + (10))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp44 = tl.load(in_ptr1 + (11))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp48 = tl.load(in_ptr1 + (12))
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK])
    tmp50 = tl.load(in_ptr1 + (13))
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK])
    tmp53 = tl.load(in_ptr1 + (14))
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK])
    tmp56 = tl.load(in_ptr1 + (15))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp4 = tmp1 + tmp3
    tmp7 = tmp4 + tmp6
    tmp10 = tmp7 + tmp9
    tmp11 = 4.0
    tmp12 = tmp10 / tmp11
    tmp17 = tmp14 + tmp16
    tmp20 = tmp17 + tmp19
    tmp23 = tmp20 + tmp22
    tmp28 = tmp25 + tmp27
    tmp31 = tmp28 + tmp30
    tmp34 = tmp31 + tmp33
    tmp35 = tmp23 + tmp34
    tmp40 = tmp37 + tmp39
    tmp43 = tmp40 + tmp42
    tmp46 = tmp43 + tmp45
    tmp47 = tmp35 + tmp46
    tmp52 = tmp49 + tmp51
    tmp55 = tmp52 + tmp54
    tmp58 = tmp55 + tmp57
    tmp59 = tmp47 + tmp58
    tmp60 = tmp59 / tmp11
    tmp61 = tmp12 - tmp60
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp61, None)
