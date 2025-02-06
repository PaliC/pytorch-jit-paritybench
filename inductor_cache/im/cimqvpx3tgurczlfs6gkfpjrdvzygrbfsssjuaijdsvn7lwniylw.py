
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_clamp_floor_mul_rsub_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_clamp_floor_mul_rsub_sub_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tmp4 = tl.load(in_ptr0 + (x0 + 16*tmp3), xmask)
    tmp5 = tl.load(in_ptr1 + (x0 + 16*tmp3 + 32*x1), xmask)
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.floor(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = 0.0
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = 3.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = tmp6 - tmp13
    tmp15 = tl_math.abs(tmp14)
    tmp16 = tmp8 - tmp15
    tmp17 = tmp1 < tmp1
    tmp18 = tl.where(tmp17, tmp1, tmp0)
    tmp19 = tl.load(in_ptr0 + (x0 + 16*tmp18), xmask)
    tmp20 = tl.load(in_ptr1 + (x0 + 16*tmp18 + 32*x1), xmask)
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.floor(tmp21)
    tmp23 = tmp22 + tmp8
    tmp24 = triton_helpers.maximum(tmp23, tmp10)
    tmp25 = triton_helpers.minimum(tmp24, tmp12)
    tmp26 = tmp21 - tmp25
    tmp27 = tl_math.abs(tmp26)
    tmp28 = tmp8 - tmp27
    tmp29 = tmp16 * tmp28
    tmp30 = triton_helpers.maximum(tmp22, tmp10)
    tmp31 = triton_helpers.minimum(tmp30, tmp12)
    tmp32 = tmp21 - tmp31
    tmp33 = tl_math.abs(tmp32)
    tmp34 = tmp8 - tmp33
    tmp35 = tmp16 * tmp34
    tmp36 = triton_helpers.maximum(tmp7, tmp10)
    tmp37 = triton_helpers.minimum(tmp36, tmp12)
    tmp38 = tmp6 - tmp37
    tmp39 = tl_math.abs(tmp38)
    tmp40 = tmp8 - tmp39
    tmp41 = tmp40 * tmp28
    tmp42 = tmp40 * tmp34
    tl.store(out_ptr0 + (x0 + 64*x1), tmp29, xmask)
    tl.store(out_ptr1 + (x0 + 64*x1), tmp35, xmask)
    tl.store(out_ptr2 + (x0 + 64*x1), tmp41, xmask)
    tl.store(out_ptr3 + (x0 + 64*x1), tmp42, xmask)
