
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sum_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 48*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr4 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (16 + x0 + 48*x1), xmask)
    tmp21 = tl.load(in_ptr1 + (1))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp24 = tl.load(in_ptr2 + (1))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp31 = tl.load(in_ptr3 + (1))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp34 = tl.load(in_ptr4 + (1))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp38 = tl.load(in_ptr0 + (32 + x0 + 48*x1), xmask)
    tmp39 = tl.load(in_ptr1 + (2))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp42 = tl.load(in_ptr2 + (2))
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK])
    tmp49 = tl.load(in_ptr3 + (2))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp52 = tl.load(in_ptr4 + (2))
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp56 = tl.load(in_ptr5 + (x0 + 48*x1), xmask)
    tmp61 = tl.load(in_ptr5 + (16 + x0 + 48*x1), xmask)
    tmp67 = tl.load(in_ptr5 + (32 + x0 + 48*x1), xmask)
    tmp3 = tmp0 - tmp2
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp16 = tmp13 * tmp15
    tmp19 = tmp16 + tmp18
    tmp23 = tmp20 - tmp22
    tmp26 = tmp25 + tmp6
    tmp27 = libdevice.sqrt(tmp26)
    tmp28 = tmp9 / tmp27
    tmp29 = tmp28 * tmp11
    tmp30 = tmp23 * tmp29
    tmp33 = tmp30 * tmp32
    tmp36 = tmp33 + tmp35
    tmp37 = tmp19 + tmp36
    tmp41 = tmp38 - tmp40
    tmp44 = tmp43 + tmp6
    tmp45 = libdevice.sqrt(tmp44)
    tmp46 = tmp9 / tmp45
    tmp47 = tmp46 * tmp11
    tmp48 = tmp41 * tmp47
    tmp51 = tmp48 * tmp50
    tmp54 = tmp51 + tmp53
    tmp55 = tmp37 + tmp54
    tmp57 = tmp56 - tmp2
    tmp58 = tmp57 * tmp12
    tmp59 = tmp58 * tmp15
    tmp60 = tmp59 + tmp18
    tmp62 = tmp61 - tmp22
    tmp63 = tmp62 * tmp29
    tmp64 = tmp63 * tmp32
    tmp65 = tmp64 + tmp35
    tmp66 = tmp60 + tmp65
    tmp68 = tmp67 - tmp40
    tmp69 = tmp68 * tmp47
    tmp70 = tmp69 * tmp50
    tmp71 = tmp70 + tmp53
    tmp72 = tmp66 + tmp71
    tl.store(out_ptr0 + (x2), tmp55, xmask)
    tl.store(out_ptr1 + (x2), tmp72, xmask)
