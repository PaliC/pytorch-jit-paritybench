
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_neg_sum_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 32, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_neg_sum_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp11 = tl.load(in_ptr0 + (4))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp13 = tl.load(in_ptr0 + (5))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + (6))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp19 = tl.load(in_ptr0 + (7))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp23 = tl.load(in_ptr0 + (8))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp25 = tl.load(in_ptr0 + (9))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp28 = tl.load(in_ptr0 + (10))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp31 = tl.load(in_ptr0 + (11))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp35 = tl.load(in_ptr0 + (12))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp37 = tl.load(in_ptr0 + (13))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK])
    tmp40 = tl.load(in_ptr0 + (14))
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
    tmp43 = tl.load(in_ptr0 + (15))
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK])
    tmp50 = tl.load(in_ptr1 + (0))
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK])
    tmp52 = tl.load(in_ptr1 + (1))
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp55 = tl.load(in_ptr1 + (2))
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK])
    tmp58 = tl.load(in_ptr1 + (3))
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK])
    tmp61 = tl.load(in_ptr1 + (4))
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK])
    tmp63 = tl.load(in_ptr1 + (5))
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK])
    tmp66 = tl.load(in_ptr1 + (6))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK])
    tmp69 = tl.load(in_ptr1 + (7))
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK])
    tmp73 = tl.load(in_ptr1 + (8))
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK])
    tmp75 = tl.load(in_ptr1 + (9))
    tmp76 = tl.broadcast_to(tmp75, [XBLOCK])
    tmp78 = tl.load(in_ptr1 + (10))
    tmp79 = tl.broadcast_to(tmp78, [XBLOCK])
    tmp81 = tl.load(in_ptr1 + (11))
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK])
    tmp85 = tl.load(in_ptr1 + (12))
    tmp86 = tl.broadcast_to(tmp85, [XBLOCK])
    tmp87 = tl.load(in_ptr1 + (13))
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK])
    tmp90 = tl.load(in_ptr1 + (14))
    tmp91 = tl.broadcast_to(tmp90, [XBLOCK])
    tmp93 = tl.load(in_ptr1 + (15))
    tmp94 = tl.broadcast_to(tmp93, [XBLOCK])
    tmp4 = tmp1 + tmp3
    tmp7 = tmp4 + tmp6
    tmp10 = tmp7 + tmp9
    tmp15 = tmp12 + tmp14
    tmp18 = tmp15 + tmp17
    tmp21 = tmp18 + tmp20
    tmp22 = tmp10 + tmp21
    tmp27 = tmp24 + tmp26
    tmp30 = tmp27 + tmp29
    tmp33 = tmp30 + tmp32
    tmp34 = tmp22 + tmp33
    tmp39 = tmp36 + tmp38
    tmp42 = tmp39 + tmp41
    tmp45 = tmp42 + tmp44
    tmp46 = tmp34 + tmp45
    tmp47 = 4.0
    tmp48 = tmp46 / tmp47
    tmp49 = -tmp48
    tmp54 = tmp51 + tmp53
    tmp57 = tmp54 + tmp56
    tmp60 = tmp57 + tmp59
    tmp65 = tmp62 + tmp64
    tmp68 = tmp65 + tmp67
    tmp71 = tmp68 + tmp70
    tmp72 = tmp60 + tmp71
    tmp77 = tmp74 + tmp76
    tmp80 = tmp77 + tmp79
    tmp83 = tmp80 + tmp82
    tmp84 = tmp72 + tmp83
    tmp89 = tmp86 + tmp88
    tmp92 = tmp89 + tmp91
    tmp95 = tmp92 + tmp94
    tmp96 = tmp84 + tmp95
    tmp97 = tmp96 / tmp47
    tmp98 = -tmp97
    tmp99 = tmp49 + tmp98
    tmp100 = 0.5
    tmp101 = tmp99 * tmp100
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp101, None)
