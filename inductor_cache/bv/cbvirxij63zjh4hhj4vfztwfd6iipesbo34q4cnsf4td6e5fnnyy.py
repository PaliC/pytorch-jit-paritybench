
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_mul_reciprocal_sqrt_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_div_mul_reciprocal_sqrt_sub_sum_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp8 = tl.load(in_ptr2 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp22 = tl.load(in_ptr0 + (1))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + (1))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp29 = tl.load(in_ptr2 + (1))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp31 = tl.load(in_ptr3 + (1))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp42 = tl.load(in_ptr0 + (2))
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK])
    tmp44 = tl.load(in_ptr1 + (2))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp49 = tl.load(in_ptr2 + (2))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp51 = tl.load(in_ptr3 + (2))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp62 = tl.load(in_ptr0 + (3))
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK])
    tmp64 = tl.load(in_ptr1 + (3))
    tmp65 = tl.broadcast_to(tmp64, [XBLOCK])
    tmp69 = tl.load(in_ptr2 + (3))
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK])
    tmp71 = tl.load(in_ptr3 + (3))
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK])
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tmp1 / tmp6
    tmp12 = tmp11 + tmp4
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = tmp9 / tmp13
    tmp15 = tmp7 - tmp14
    tmp16 = tl_math.abs(tmp15)
    tmp17 = 1.0
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full([1], 1, tl.int32)
    tmp20 = tmp19 / tmp18
    tmp21 = tmp20 * tmp17
    tmp26 = tmp25 + tmp4
    tmp27 = libdevice.sqrt(tmp26)
    tmp28 = tmp23 / tmp27
    tmp33 = tmp32 + tmp4
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp30 / tmp34
    tmp36 = tmp28 - tmp35
    tmp37 = tl_math.abs(tmp36)
    tmp38 = tmp37 + tmp17
    tmp39 = tmp19 / tmp38
    tmp40 = tmp39 * tmp17
    tmp41 = tmp21 + tmp40
    tmp46 = tmp45 + tmp4
    tmp47 = libdevice.sqrt(tmp46)
    tmp48 = tmp43 / tmp47
    tmp53 = tmp52 + tmp4
    tmp54 = libdevice.sqrt(tmp53)
    tmp55 = tmp50 / tmp54
    tmp56 = tmp48 - tmp55
    tmp57 = tl_math.abs(tmp56)
    tmp58 = tmp57 + tmp17
    tmp59 = tmp19 / tmp58
    tmp60 = tmp59 * tmp17
    tmp61 = tmp41 + tmp60
    tmp66 = tmp65 + tmp4
    tmp67 = libdevice.sqrt(tmp66)
    tmp68 = tmp63 / tmp67
    tmp73 = tmp72 + tmp4
    tmp74 = libdevice.sqrt(tmp73)
    tmp75 = tmp70 / tmp74
    tmp76 = tmp68 - tmp75
    tmp77 = tl_math.abs(tmp76)
    tmp78 = tmp77 + tmp17
    tmp79 = tmp19 / tmp78
    tmp80 = tmp79 * tmp17
    tmp81 = tmp61 + tmp80
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp81, None)
