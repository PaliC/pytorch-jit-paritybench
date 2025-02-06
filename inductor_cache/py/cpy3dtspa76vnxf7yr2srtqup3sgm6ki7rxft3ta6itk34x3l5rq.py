
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 12)
    x0 = (xindex % 4)
    x2 = xindex // 48
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (16 + x0 + 4*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr0 + (32 + x0 + 4*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = 1.7320508075688772
    tmp9 = tmp7 * tmp8
    tmp10 = tl.load(in_ptr0 + (x0 + 4*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp11 = 2.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 - tmp5
    tmp14 = tmp13 - tmp6
    tmp15 = libdevice.atan2(tmp9, tmp14)
    tmp16 = 6.283185307179586
    tmp17 = tmp15 % tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = tmp17 != tmp18
    tmp20 = (libdevice.signbit(tmp17) != 0) if (tmp17).dtype is tl.float32 else tmp17 < 0
    tmp21 = (libdevice.signbit(tmp16) != 0) if (tmp16).dtype is tl.float32 else tmp16 < 0
    tmp22 = tmp20 != tmp21
    tmp23 = tmp19 & tmp22
    tmp24 = tmp17 + tmp16
    tmp25 = tl.where(tmp23, tmp24, tmp17)
    tmp26 = 0.15915494309189535
    tmp27 = tmp25 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp4, tmp27, tmp28)
    tmp30 = tmp0 >= tmp3
    tmp31 = tl.full([1], 8, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tmp30 & tmp32
    tmp34 = tl.load(in_ptr0 + (x0 + 4*((-4) + x1) + 64*x2), tmp33 & xmask, other=0.0)
    tmp35 = tl.load(in_ptr0 + (16 + x0 + 4*((-4) + x1) + 64*x2), tmp33 & xmask, other=0.0)
    tmp36 = triton_helpers.minimum(tmp34, tmp35)
    tmp37 = tl.load(in_ptr0 + (32 + x0 + 4*((-4) + x1) + 64*x2), tmp33 & xmask, other=0.0)
    tmp38 = triton_helpers.minimum(tmp36, tmp37)
    tmp39 = tl.load(in_ptr0 + (48 + x0 + 4*((-4) + x1) + 64*x2), tmp33 & xmask, other=0.0)
    tmp40 = triton_helpers.minimum(tmp38, tmp39)
    tmp41 = triton_helpers.maximum(tmp34, tmp35)
    tmp42 = triton_helpers.maximum(tmp41, tmp37)
    tmp43 = triton_helpers.maximum(tmp42, tmp39)
    tmp44 = 1e-08
    tmp45 = tmp43 + tmp44
    tmp46 = tmp40 / tmp45
    tmp47 = 1.0
    tmp48 = tmp47 - tmp46
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp33, tmp48, tmp49)
    tmp51 = tmp0 >= tmp31
    tmp52 = tl.full([1], 12, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tl.load(in_ptr0 + (x0 + 4*((-8) + x1) + 64*x2), tmp51 & xmask, other=0.0)
    tmp55 = tl.load(in_ptr0 + (16 + x0 + 4*((-8) + x1) + 64*x2), tmp51 & xmask, other=0.0)
    tmp56 = triton_helpers.maximum(tmp54, tmp55)
    tmp57 = tl.load(in_ptr0 + (32 + x0 + 4*((-8) + x1) + 64*x2), tmp51 & xmask, other=0.0)
    tmp58 = triton_helpers.maximum(tmp56, tmp57)
    tmp59 = tl.load(in_ptr0 + (48 + x0 + 4*((-8) + x1) + 64*x2), tmp51 & xmask, other=0.0)
    tmp60 = triton_helpers.maximum(tmp58, tmp59)
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp51, tmp60, tmp61)
    tmp63 = tl.where(tmp33, tmp50, tmp62)
    tmp64 = tl.where(tmp4, tmp29, tmp63)
    tl.store(out_ptr0 + (x3), tmp64, xmask)
