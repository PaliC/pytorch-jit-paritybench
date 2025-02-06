
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'out_ptr21': '*fp32', 'out_ptr22': '*fp32', 'out_ptr23': '*fp32', 'out_ptr24': '*fp32', 'out_ptr25': '*fp32', 'out_ptr26': '*fp32', 'out_ptr27': '*fp32', 'out_ptr28': '*fp32', 'out_ptr29': '*fp32', 'out_ptr30': '*fp32', 'out_ptr31': '*fp32', 'out_ptr32': '*fp32', 'out_ptr33': '*fp32', 'out_ptr34': '*fp32', 'out_ptr35': '*fp32', 'out_ptr36': '*fp32', 'out_ptr37': '*fp32', 'out_ptr38': '*fp32', 'out_ptr39': '*fp32', 'out_ptr40': '*fp32', 'out_ptr41': '*fp32', 'out_ptr42': '*fp32', 'out_ptr43': '*fp32', 'out_ptr44': '*fp32', 'out_ptr45': '*fp32', 'out_ptr46': '*fp32', 'out_ptr47': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 17, 33, 49), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35, out_ptr36, out_ptr37, out_ptr38, out_ptr39, out_ptr40, out_ptr41, out_ptr42, out_ptr43, out_ptr44, out_ptr45, out_ptr46, out_ptr47, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp5 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp9 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp1 = -1.0
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp6 = -1.5
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tmp10 = tmp9 * tmp6
    tmp11 = tmp8 + tmp10
    tmp12 = -0.5
    tmp13 = tmp9 * tmp12
    tmp14 = tmp8 + tmp13
    tmp15 = 0.5
    tmp16 = tmp9 * tmp15
    tmp17 = tmp8 + tmp16
    tmp18 = 1.5
    tmp19 = tmp9 * tmp18
    tmp20 = tmp8 + tmp19
    tmp21 = tmp5 * tmp12
    tmp22 = tmp4 + tmp21
    tmp23 = tmp22 + tmp10
    tmp24 = tmp22 + tmp13
    tmp25 = tmp22 + tmp16
    tmp26 = tmp22 + tmp19
    tmp27 = tmp5 * tmp15
    tmp28 = tmp4 + tmp27
    tmp29 = tmp28 + tmp10
    tmp30 = tmp28 + tmp13
    tmp31 = tmp28 + tmp16
    tmp32 = tmp28 + tmp19
    tmp33 = tmp5 * tmp18
    tmp34 = tmp4 + tmp33
    tmp35 = tmp34 + tmp10
    tmp36 = tmp34 + tmp13
    tmp37 = tmp34 + tmp16
    tmp38 = tmp34 + tmp19
    tmp39 = 0.0
    tmp40 = tmp0 + tmp39
    tmp41 = tmp40 * tmp3
    tmp42 = tmp41 + tmp7
    tmp43 = tmp42 + tmp10
    tmp44 = tmp42 + tmp13
    tmp45 = tmp42 + tmp16
    tmp46 = tmp42 + tmp19
    tmp47 = tmp41 + tmp21
    tmp48 = tmp47 + tmp10
    tmp49 = tmp47 + tmp13
    tmp50 = tmp47 + tmp16
    tmp51 = tmp47 + tmp19
    tmp52 = tmp41 + tmp27
    tmp53 = tmp52 + tmp10
    tmp54 = tmp52 + tmp13
    tmp55 = tmp52 + tmp16
    tmp56 = tmp52 + tmp19
    tmp57 = tmp41 + tmp33
    tmp58 = tmp57 + tmp10
    tmp59 = tmp57 + tmp13
    tmp60 = tmp57 + tmp16
    tmp61 = tmp57 + tmp19
    tmp62 = tmp0 + tmp3
    tmp63 = tmp62 * tmp3
    tmp64 = tmp63 + tmp7
    tmp65 = tmp64 + tmp10
    tmp66 = tmp64 + tmp13
    tmp67 = tmp64 + tmp16
    tmp68 = tmp64 + tmp19
    tmp69 = tmp63 + tmp21
    tmp70 = tmp69 + tmp10
    tmp71 = tmp69 + tmp13
    tmp72 = tmp69 + tmp16
    tmp73 = tmp69 + tmp19
    tmp74 = tmp63 + tmp27
    tmp75 = tmp74 + tmp10
    tmp76 = tmp74 + tmp13
    tmp77 = tmp74 + tmp16
    tmp78 = tmp74 + tmp19
    tmp79 = tmp63 + tmp33
    tmp80 = tmp79 + tmp10
    tmp81 = tmp79 + tmp13
    tmp82 = tmp79 + tmp16
    tmp83 = tmp79 + tmp19
    tl.store(out_ptr0 + (16*x2), tmp11, xmask)
    tl.store(out_ptr1 + (16*x2), tmp14, xmask)
    tl.store(out_ptr2 + (16*x2), tmp17, xmask)
    tl.store(out_ptr3 + (16*x2), tmp20, xmask)
    tl.store(out_ptr4 + (16*x2), tmp23, xmask)
    tl.store(out_ptr5 + (16*x2), tmp24, xmask)
    tl.store(out_ptr6 + (16*x2), tmp25, xmask)
    tl.store(out_ptr7 + (16*x2), tmp26, xmask)
    tl.store(out_ptr8 + (16*x2), tmp29, xmask)
    tl.store(out_ptr9 + (16*x2), tmp30, xmask)
    tl.store(out_ptr10 + (16*x2), tmp31, xmask)
    tl.store(out_ptr11 + (16*x2), tmp32, xmask)
    tl.store(out_ptr12 + (16*x2), tmp35, xmask)
    tl.store(out_ptr13 + (16*x2), tmp36, xmask)
    tl.store(out_ptr14 + (16*x2), tmp37, xmask)
    tl.store(out_ptr15 + (16*x2), tmp38, xmask)
    tl.store(out_ptr16 + (16*x2), tmp43, xmask)
    tl.store(out_ptr17 + (16*x2), tmp44, xmask)
    tl.store(out_ptr18 + (16*x2), tmp45, xmask)
    tl.store(out_ptr19 + (16*x2), tmp46, xmask)
    tl.store(out_ptr20 + (16*x2), tmp48, xmask)
    tl.store(out_ptr21 + (16*x2), tmp49, xmask)
    tl.store(out_ptr22 + (16*x2), tmp50, xmask)
    tl.store(out_ptr23 + (16*x2), tmp51, xmask)
    tl.store(out_ptr24 + (16*x2), tmp53, xmask)
    tl.store(out_ptr25 + (16*x2), tmp54, xmask)
    tl.store(out_ptr26 + (16*x2), tmp55, xmask)
    tl.store(out_ptr27 + (16*x2), tmp56, xmask)
    tl.store(out_ptr28 + (16*x2), tmp58, xmask)
    tl.store(out_ptr29 + (16*x2), tmp59, xmask)
    tl.store(out_ptr30 + (16*x2), tmp60, xmask)
    tl.store(out_ptr31 + (16*x2), tmp61, xmask)
    tl.store(out_ptr32 + (16*x2), tmp65, xmask)
    tl.store(out_ptr33 + (16*x2), tmp66, xmask)
    tl.store(out_ptr34 + (16*x2), tmp67, xmask)
    tl.store(out_ptr35 + (16*x2), tmp68, xmask)
    tl.store(out_ptr36 + (16*x2), tmp70, xmask)
    tl.store(out_ptr37 + (16*x2), tmp71, xmask)
    tl.store(out_ptr38 + (16*x2), tmp72, xmask)
    tl.store(out_ptr39 + (16*x2), tmp73, xmask)
    tl.store(out_ptr40 + (16*x2), tmp75, xmask)
    tl.store(out_ptr41 + (16*x2), tmp76, xmask)
    tl.store(out_ptr42 + (16*x2), tmp77, xmask)
    tl.store(out_ptr43 + (16*x2), tmp78, xmask)
    tl.store(out_ptr44 + (16*x2), tmp80, xmask)
    tl.store(out_ptr45 + (16*x2), tmp81, xmask)
    tl.store(out_ptr46 + (16*x2), tmp82, xmask)
    tl.store(out_ptr47 + (16*x2), tmp83, xmask)
