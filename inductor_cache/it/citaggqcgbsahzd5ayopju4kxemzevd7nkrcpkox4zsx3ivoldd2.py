
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
    triton_meta={'signature': {'in_ptr0': 'i64', 'in_ptr1': 'fp64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (2, 3, 4, 5), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_exp_lift_fresh_log_mul_pow_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_exp_lift_fresh_log_mul_pow_sub_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = in_ptr0
    tmp7 = in_ptr1
    tmp9 = tl.load(in_ptr2 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp11 = tl.load(in_ptr3 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp17 = tl.load(in_ptr2 + (1))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp19 = tl.load(in_ptr3 + (1))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp25 = tl.load(in_ptr2 + (2))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp27 = tl.load(in_ptr3 + (2))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp33 = tl.load(in_ptr2 + (3))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp35 = tl.load(in_ptr3 + (3))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp43 = tl.load(in_ptr4 + (0))
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK])
    tmp46 = tl.load(in_ptr4 + (1))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp50 = tl.load(in_ptr4 + (2))
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK])
    tmp54 = tl.load(in_ptr4 + (3))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp1 = tmp0.to(tl.float64)
    tmp2 = tl.full([1], -0.5, tl.float64)
    tmp3 = tmp2 * tmp1
    tmp4 = tl.full([1], 1.8378770664093453, tl.float64)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp13 = 3.0
    tmp14 = tmp12 * tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tmp10 * tmp15
    tmp21 = tmp20 * tmp13
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp18 * tmp22
    tmp24 = tmp16 + tmp23
    tmp29 = tmp28 * tmp13
    tmp30 = tl_math.exp(tmp29)
    tmp31 = tmp26 * tmp30
    tmp32 = tmp24 + tmp31
    tmp37 = tmp36 * tmp13
    tmp38 = tl_math.exp(tmp37)
    tmp39 = tmp34 * tmp38
    tmp40 = tmp32 + tmp39
    tmp41 = tmp8 * tmp40
    tmp42 = tmp6 - tmp41
    tmp45 = tmp44 * tmp44
    tmp48 = tmp47 * tmp47
    tmp49 = tmp45 + tmp48
    tmp52 = tmp51 * tmp51
    tmp53 = tmp49 + tmp52
    tmp56 = tmp55 * tmp55
    tmp57 = tmp53 + tmp56
    tmp58 = 0.5
    tmp59 = tmp57 * tmp58
    tmp60 = tmp42 - tmp59
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp60, None)
