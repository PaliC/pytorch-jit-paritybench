
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
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + (1))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp18 = tl.load(in_ptr1 + (1))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp21 = tl.load(in_ptr2 + (1))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp31 = tl.load(in_ptr0 + (2))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp33 = tl.load(in_ptr1 + (2))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp36 = tl.load(in_ptr2 + (2))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp46 = tl.load(in_ptr0 + (3))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp48 = tl.load(in_ptr1 + (3))
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK])
    tmp51 = tl.load(in_ptr2 + (3))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp4 = tmp1 + tmp3
    tmp7 = tmp4 + tmp6
    tmp8 = 0.0
    tmp9 = tmp7 == tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 1e-07
    tmp12 = tmp10 * tmp11
    tmp13 = tmp1 + tmp12
    tmp14 = tmp7 + tmp11
    tmp15 = tmp13 / tmp14
    tmp20 = tmp17 + tmp19
    tmp23 = tmp20 + tmp22
    tmp24 = tmp23 == tmp8
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp11
    tmp27 = tmp17 + tmp26
    tmp28 = tmp23 + tmp11
    tmp29 = tmp27 / tmp28
    tmp30 = tmp15 + tmp29
    tmp35 = tmp32 + tmp34
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38 == tmp8
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp40 * tmp11
    tmp42 = tmp32 + tmp41
    tmp43 = tmp38 + tmp11
    tmp44 = tmp42 / tmp43
    tmp45 = tmp30 + tmp44
    tmp50 = tmp47 + tmp49
    tmp53 = tmp50 + tmp52
    tmp54 = tmp53 == tmp8
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp55 * tmp11
    tmp57 = tmp47 + tmp56
    tmp58 = tmp53 + tmp11
    tmp59 = tmp57 / tmp58
    tmp60 = tmp45 + tmp59
    tmp61 = 4.0
    tmp62 = tmp60 / tmp61
    tmp63 = 1.0
    tmp64 = tmp63 - tmp62
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp64, None)
