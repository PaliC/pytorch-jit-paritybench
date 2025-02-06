
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_eq_mean_mul_rsub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_eq_mean_mul_rsub_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (1))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp23 = tl.load(in_ptr1 + (1))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp26 = tl.load(in_ptr2 + (1))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp38 = tl.load(in_ptr0 + (2))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp41 = tl.load(in_ptr1 + (2))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp44 = tl.load(in_ptr2 + (2))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp56 = tl.load(in_ptr0 + (3))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp59 = tl.load(in_ptr1 + (3))
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK])
    tmp62 = tl.load(in_ptr2 + (3))
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK])
    tmp2 = 2.0
    tmp3 = tmp1 * tmp2
    tmp6 = tmp1 + tmp5
    tmp9 = tmp6 + tmp8
    tmp10 = 0.0
    tmp11 = tmp9 == tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 1e-07
    tmp14 = tmp12 * tmp13
    tmp15 = tmp3 + tmp14
    tmp16 = tmp3 + tmp5
    tmp17 = tmp16 + tmp8
    tmp18 = tmp17 + tmp13
    tmp19 = tmp15 / tmp18
    tmp22 = tmp21 * tmp2
    tmp25 = tmp21 + tmp24
    tmp28 = tmp25 + tmp27
    tmp29 = tmp28 == tmp10
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp13
    tmp32 = tmp22 + tmp31
    tmp33 = tmp22 + tmp24
    tmp34 = tmp33 + tmp27
    tmp35 = tmp34 + tmp13
    tmp36 = tmp32 / tmp35
    tmp37 = tmp19 + tmp36
    tmp40 = tmp39 * tmp2
    tmp43 = tmp39 + tmp42
    tmp46 = tmp43 + tmp45
    tmp47 = tmp46 == tmp10
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp48 * tmp13
    tmp50 = tmp40 + tmp49
    tmp51 = tmp40 + tmp42
    tmp52 = tmp51 + tmp45
    tmp53 = tmp52 + tmp13
    tmp54 = tmp50 / tmp53
    tmp55 = tmp37 + tmp54
    tmp58 = tmp57 * tmp2
    tmp61 = tmp57 + tmp60
    tmp64 = tmp61 + tmp63
    tmp65 = tmp64 == tmp10
    tmp66 = tmp65.to(tl.float32)
    tmp67 = tmp66 * tmp13
    tmp68 = tmp58 + tmp67
    tmp69 = tmp58 + tmp60
    tmp70 = tmp69 + tmp63
    tmp71 = tmp70 + tmp13
    tmp72 = tmp68 / tmp71
    tmp73 = tmp55 + tmp72
    tmp74 = 4.0
    tmp75 = tmp73 / tmp74
    tmp76 = 1.0
    tmp77 = tmp76 - tmp75
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp77, None)
