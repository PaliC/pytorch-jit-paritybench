
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': (7,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_rsub_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_rsub_sub_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr3 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp12 = tl.load(in_ptr5 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp22 = tl.load(in_ptr0 + (1))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + (1))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp27 = tl.load(in_ptr2 + (1))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp30 = tl.load(in_ptr3 + (1))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp32 = tl.load(in_ptr4 + (1))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp34 = tl.load(in_ptr5 + (1))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp44 = tl.load(in_ptr0 + (2))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp46 = tl.load(in_ptr1 + (2))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp49 = tl.load(in_ptr2 + (2))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp52 = tl.load(in_ptr3 + (2))
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp54 = tl.load(in_ptr4 + (2))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp56 = tl.load(in_ptr5 + (2))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp66 = tl.load(in_ptr0 + (3))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK])
    tmp68 = tl.load(in_ptr1 + (3))
    tmp69 = tl.broadcast_to(tmp68, [XBLOCK])
    tmp71 = tl.load(in_ptr2 + (3))
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK])
    tmp74 = tl.load(in_ptr3 + (3))
    tmp75 = tl.broadcast_to(tmp74, [XBLOCK])
    tmp76 = tl.load(in_ptr4 + (3))
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK])
    tmp78 = tl.load(in_ptr5 + (3))
    tmp79 = tl.broadcast_to(tmp78, [XBLOCK])
    tmp4 = tmp1 / tmp3
    tmp7 = tmp4 - tmp6
    tmp14 = tmp11 / tmp13
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tmp16 + tmp9
    tmp18 = tmp9 / tmp17
    tmp19 = tmp18 * tmp9
    tmp20 = tmp7 - tmp19
    tmp21 = tmp15 - tmp20
    tmp26 = tmp23 / tmp25
    tmp29 = tmp26 - tmp28
    tmp36 = tmp33 / tmp35
    tmp37 = tmp15 - tmp36
    tmp38 = tmp37 + tmp31
    tmp39 = tmp31 / tmp38
    tmp40 = tmp39 * tmp31
    tmp41 = tmp29 - tmp40
    tmp42 = tmp15 - tmp41
    tmp43 = tmp21 + tmp42
    tmp48 = tmp45 / tmp47
    tmp51 = tmp48 - tmp50
    tmp58 = tmp55 / tmp57
    tmp59 = tmp15 - tmp58
    tmp60 = tmp59 + tmp53
    tmp61 = tmp53 / tmp60
    tmp62 = tmp61 * tmp53
    tmp63 = tmp51 - tmp62
    tmp64 = tmp15 - tmp63
    tmp65 = tmp43 + tmp64
    tmp70 = tmp67 / tmp69
    tmp73 = tmp70 - tmp72
    tmp80 = tmp77 / tmp79
    tmp81 = tmp15 - tmp80
    tmp82 = tmp81 + tmp75
    tmp83 = tmp75 / tmp82
    tmp84 = tmp83 * tmp75
    tmp85 = tmp73 - tmp84
    tmp86 = tmp15 - tmp85
    tmp87 = tmp65 + tmp86
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp87, None)
