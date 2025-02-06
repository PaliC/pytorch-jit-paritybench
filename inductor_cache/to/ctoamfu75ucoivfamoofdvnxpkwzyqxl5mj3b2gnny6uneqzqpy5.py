
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_softplus_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_softplus_sum_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16
    x0 = (xindex % 16)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1)), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-4) + x1)), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 12, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 16*((-8) + x1)), tmp14 & xmask, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 16, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x0 + 16*((-12) + x1)), tmp16 & xmask, other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp23 = 4 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tl.load(in_ptr0 + (x0 + 16*(4 + x1)), tmp25 & xmask, other=0.0)
    tmp27 = tmp23 >= tmp3
    tmp28 = tmp23 < tmp7
    tmp29 = tmp27 & tmp28
    tmp30 = tl.load(in_ptr1 + (x0 + 16*(x1)), tmp29 & xmask, other=0.0)
    tmp31 = tmp23 >= tmp7
    tmp32 = tmp23 < tmp12
    tmp33 = tmp31 & tmp32
    tmp34 = tl.load(in_ptr2 + (x0 + 16*((-4) + x1)), tmp33 & xmask, other=0.0)
    tmp35 = tmp23 >= tmp12
    tmp36 = tmp23 < tmp17
    tmp37 = tl.load(in_ptr3 + (x0 + 16*((-8) + x1)), tmp35 & xmask, other=0.0)
    tmp38 = tl.where(tmp33, tmp34, tmp37)
    tmp39 = tl.where(tmp29, tmp30, tmp38)
    tmp40 = tl.where(tmp25, tmp26, tmp39)
    tmp41 = tmp22 + tmp40
    tmp42 = 8 + x1
    tmp43 = tmp42 >= tmp1
    tmp44 = tmp42 < tmp3
    tmp45 = tl.load(in_ptr0 + (x0 + 16*(8 + x1)), tmp44 & xmask, other=0.0)
    tmp46 = tmp42 >= tmp3
    tmp47 = tmp42 < tmp7
    tmp48 = tmp46 & tmp47
    tmp49 = tl.load(in_ptr1 + (x0 + 16*(4 + x1)), tmp48 & xmask, other=0.0)
    tmp50 = tmp42 >= tmp7
    tmp51 = tmp42 < tmp12
    tmp52 = tmp50 & tmp51
    tmp53 = tl.load(in_ptr2 + (x0 + 16*(x1)), tmp52 & xmask, other=0.0)
    tmp54 = tmp42 >= tmp12
    tmp55 = tmp42 < tmp17
    tmp56 = tl.load(in_ptr3 + (x0 + 16*((-4) + x1)), tmp54 & xmask, other=0.0)
    tmp57 = tl.where(tmp52, tmp53, tmp56)
    tmp58 = tl.where(tmp48, tmp49, tmp57)
    tmp59 = tl.where(tmp44, tmp45, tmp58)
    tmp60 = tmp41 + tmp59
    tmp61 = 12 + x1
    tmp62 = tmp61 >= tmp1
    tmp63 = tmp61 < tmp3
    tmp64 = tl.load(in_ptr0 + (x0 + 16*(12 + x1)), tmp63 & xmask, other=0.0)
    tmp65 = tmp61 >= tmp3
    tmp66 = tmp61 < tmp7
    tmp67 = tmp65 & tmp66
    tmp68 = tl.load(in_ptr1 + (x0 + 16*(8 + x1)), tmp67 & xmask, other=0.0)
    tmp69 = tmp61 >= tmp7
    tmp70 = tmp61 < tmp12
    tmp71 = tmp69 & tmp70
    tmp72 = tl.load(in_ptr2 + (x0 + 16*(4 + x1)), tmp71 & xmask, other=0.0)
    tmp73 = tmp61 >= tmp12
    tmp74 = tmp61 < tmp17
    tmp75 = tl.load(in_ptr3 + (x0 + 16*(x1)), tmp73 & xmask, other=0.0)
    tmp76 = tl.where(tmp71, tmp72, tmp75)
    tmp77 = tl.where(tmp67, tmp68, tmp76)
    tmp78 = tl.where(tmp63, tmp64, tmp77)
    tmp79 = tmp60 + tmp78
    tmp80 = 1.0
    tmp81 = tmp79 * tmp80
    tmp82 = 20.0
    tmp83 = tmp81 > tmp82
    tmp84 = tl_math.exp(tmp81)
    tmp85 = libdevice.log1p(tmp84)
    tmp86 = tmp85 * tmp80
    tmp87 = tl.where(tmp83, tmp79, tmp86)
    tl.store(out_ptr0 + (x2), tmp81, xmask)
    tl.store(in_out_ptr0 + (x2), tmp87, xmask)
