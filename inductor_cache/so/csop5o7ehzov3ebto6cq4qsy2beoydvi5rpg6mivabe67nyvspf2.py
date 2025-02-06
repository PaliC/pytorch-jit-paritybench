
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': (8,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_exp_logsumexp_mean_mul_neg_sub_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_exp_logsumexp_mean_mul_neg_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (1))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp14 = tl.load(in_ptr1 + (1))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp21 = tl.load(in_ptr0 + (2))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + (2))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp31 = tl.load(in_ptr0 + (3))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp34 = tl.load(in_ptr1 + (3))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp43 = tl.load(in_ptr2 + (0))
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK])
    tmp45 = tl.load(in_ptr3 + (0))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp50 = tl.load(in_ptr4 + (0))
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK])
    tmp54 = tl.load(in_ptr2 + (1))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp57 = tl.load(in_ptr4 + (1))
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK])
    tmp62 = tl.load(in_ptr2 + (2))
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK])
    tmp65 = tl.load(in_ptr4 + (2))
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK])
    tmp70 = tl.load(in_ptr2 + (3))
    tmp71 = tl.broadcast_to(tmp70, [XBLOCK])
    tmp73 = tl.load(in_ptr4 + (3))
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK])
    tmp2 = tl_math.log(tmp1)
    tmp5 = tl_math.abs(tmp4)
    tmp6 = float("inf")
    tmp7 = tmp5 == tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp2 + tmp9
    tmp13 = tl_math.log(tmp12)
    tmp16 = tl_math.abs(tmp15)
    tmp17 = tmp16 == tmp6
    tmp18 = tl.where(tmp17, tmp8, tmp15)
    tmp19 = tmp13 + tmp18
    tmp20 = tmp10 + tmp19
    tmp23 = tl_math.log(tmp22)
    tmp26 = tl_math.abs(tmp25)
    tmp27 = tmp26 == tmp6
    tmp28 = tl.where(tmp27, tmp8, tmp25)
    tmp29 = tmp23 + tmp28
    tmp30 = tmp20 + tmp29
    tmp33 = tl_math.log(tmp32)
    tmp36 = tl_math.abs(tmp35)
    tmp37 = tmp36 == tmp6
    tmp38 = tl.where(tmp37, tmp8, tmp35)
    tmp39 = tmp33 + tmp38
    tmp40 = tmp30 + tmp39
    tmp41 = 4.0
    tmp42 = tmp40 / tmp41
    tmp47 = tl_math.exp(tmp46)
    tmp48 = triton_helpers.minimum(tmp47, tmp6)
    tmp49 = tmp44 * tmp48
    tmp52 = tmp49 - tmp51
    tmp53 = -tmp52
    tmp56 = tmp55 * tmp48
    tmp59 = tmp56 - tmp58
    tmp60 = -tmp59
    tmp61 = tmp53 + tmp60
    tmp64 = tmp63 * tmp48
    tmp67 = tmp64 - tmp66
    tmp68 = -tmp67
    tmp69 = tmp61 + tmp68
    tmp72 = tmp71 * tmp48
    tmp75 = tmp72 - tmp74
    tmp76 = -tmp75
    tmp77 = tmp69 + tmp76
    tmp78 = tmp77 / tmp41
    tmp79 = tmp51 + tmp58
    tmp80 = tmp79 + tmp66
    tmp81 = tmp80 + tmp74
    tmp82 = tmp81 / tmp41
    tmp83 = tmp78 - tmp82
    tmp84 = tmp42 + tmp82
    tmp85 = tmp78 + tmp42
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp83, None)
    tl.store(out_ptr3 + (tl.full([XBLOCK], 0, tl.int32)), tmp84, None)
    tl.store(out_ptr4 + (tl.full([XBLOCK], 0, tl.int32)), tmp85, None)
