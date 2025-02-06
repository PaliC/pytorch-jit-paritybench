
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp7 = -1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.load(in_ptr2 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp11 = tmp10 * tmp7
    tmp12 = tmp9 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 8, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr0 + (x0 + 16*((-4) + x1) + 64*x2), tmp18 & xmask, other=0.0)
    tmp20 = tl.load(in_ptr1 + (x0 + 16*((-4) + x1) + 64*x2), tmp18 & xmask, other=0.0)
    tmp21 = -1.0
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tl.load(in_ptr2 + (x0 + 16*((-4) + x1) + 64*x2), tmp18 & xmask, other=0.0)
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp18, tmp27, tmp28)
    tmp30 = tmp0 >= tmp16
    tmp31 = tl.full([1], 12, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tmp30 & tmp32
    tmp34 = tl.load(in_ptr0 + (x0 + 16*((-8) + x1) + 64*x2), tmp33 & xmask, other=0.0)
    tmp35 = tl.load(in_ptr1 + (x0 + 16*((-8) + x1) + 64*x2), tmp33 & xmask, other=0.0)
    tmp36 = 1.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp34 + tmp37
    tmp39 = tl.load(in_ptr2 + (x0 + 16*((-8) + x1) + 64*x2), tmp33 & xmask, other=0.0)
    tmp40 = -1.0
    tmp41 = tmp39 * tmp40
    tmp42 = tmp38 + tmp41
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp33, tmp42, tmp43)
    tmp45 = tmp0 >= tmp31
    tmp46 = tl.full([1], 16, tl.int64)
    tmp47 = tmp0 < tmp46
    tmp48 = tl.load(in_ptr0 + (x0 + 16*((-12) + x1) + 64*x2), tmp45 & xmask, other=0.0)
    tmp49 = tl.load(in_ptr1 + (x0 + 16*((-12) + x1) + 64*x2), tmp45 & xmask, other=0.0)
    tmp50 = 1.0
    tmp51 = tmp49 * tmp50
    tmp52 = tmp48 + tmp51
    tmp53 = tl.load(in_ptr2 + (x0 + 16*((-12) + x1) + 64*x2), tmp45 & xmask, other=0.0)
    tmp54 = tmp53 * tmp50
    tmp55 = tmp52 + tmp54
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp45, tmp55, tmp56)
    tmp58 = tl.where(tmp33, tmp44, tmp57)
    tmp59 = tl.where(tmp18, tmp29, tmp58)
    tmp60 = tl.where(tmp4, tmp14, tmp59)
    tl.store(out_ptr0 + (x3), tmp60, xmask)
