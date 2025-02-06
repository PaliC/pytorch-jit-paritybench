
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_convolution_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_convolution_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp5 = tl.load(in_ptr1 + (1))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp10 = tl.load(in_ptr1 + (2))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp14 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp15 = tl.load(in_ptr1 + (3))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp30 = tl.load(in_ptr2 + (x0 + 64*x1), xmask)
    tmp32 = tl.load(in_ptr2 + (16 + x0 + 64*x1), xmask)
    tmp35 = tl.load(in_ptr2 + (32 + x0 + 64*x1), xmask)
    tmp38 = tl.load(in_ptr2 + (48 + x0 + 64*x1), xmask)
    tmp3 = tmp0 + tmp2
    tmp7 = tmp4 + tmp6
    tmp8 = triton_helpers.maximum(tmp3, tmp7)
    tmp12 = tmp9 + tmp11
    tmp13 = triton_helpers.maximum(tmp8, tmp12)
    tmp17 = tmp14 + tmp16
    tmp18 = triton_helpers.maximum(tmp13, tmp17)
    tmp19 = tmp3 - tmp18
    tmp20 = tl_math.exp(tmp19)
    tmp21 = tmp7 - tmp18
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp20 + tmp22
    tmp24 = tmp12 - tmp18
    tmp25 = tl_math.exp(tmp24)
    tmp26 = tmp23 + tmp25
    tmp27 = tmp17 - tmp18
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tmp26 + tmp28
    tmp31 = tmp30 + tmp2
    tmp33 = tmp32 + tmp6
    tmp34 = triton_helpers.maximum(tmp31, tmp33)
    tmp36 = tmp35 + tmp11
    tmp37 = triton_helpers.maximum(tmp34, tmp36)
    tmp39 = tmp38 + tmp16
    tmp40 = triton_helpers.maximum(tmp37, tmp39)
    tmp41 = tmp31 - tmp40
    tmp42 = tl_math.exp(tmp41)
    tmp43 = tmp33 - tmp40
    tmp44 = tl_math.exp(tmp43)
    tmp45 = tmp42 + tmp44
    tmp46 = tmp36 - tmp40
    tmp47 = tl_math.exp(tmp46)
    tmp48 = tmp45 + tmp47
    tmp49 = tmp39 - tmp40
    tmp50 = tl_math.exp(tmp49)
    tmp51 = tmp48 + tmp50
    tl.store(out_ptr0 + (x2), tmp18, xmask)
    tl.store(out_ptr1 + (x2), tmp29, xmask)
    tl.store(out_ptr2 + (x2), tmp40, xmask)
    tl.store(out_ptr3 + (x2), tmp51, xmask)
