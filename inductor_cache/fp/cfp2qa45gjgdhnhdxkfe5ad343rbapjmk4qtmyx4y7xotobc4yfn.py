
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*i1', 'in_ptr3': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_mul_sum_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_exp_mul_sum_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (0)).to(tl.int1)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp8 = tl.load(in_ptr1 + (4 + x0), xmask)
    tmp10 = tl.load(in_ptr2 + (1)).to(tl.int1)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp16 = tl.load(in_ptr1 + (8 + x0), xmask)
    tmp18 = tl.load(in_ptr2 + (2)).to(tl.int1)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + (12 + x0), xmask)
    tmp26 = tl.load(in_ptr2 + (3)).to(tl.int1)
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp33 = tl.load(in_ptr3 + (x0), xmask)
    tmp35 = tl.load(in_ptr3 + (4 + x0), xmask)
    tmp38 = tl.load(in_ptr3 + (8 + x0), xmask)
    tmp41 = tl.load(in_ptr3 + (12 + x0), xmask)
    tmp2 = libdevice.exp(tmp1)
    tmp5 = tmp4.to(tl.int64)
    tmp6 = tmp5.to(tl.float64)
    tmp7 = tmp2 * tmp6
    tmp9 = libdevice.exp(tmp8)
    tmp12 = tmp11.to(tl.int64)
    tmp13 = tmp12.to(tl.float64)
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 + tmp14
    tmp17 = libdevice.exp(tmp16)
    tmp20 = tmp19.to(tl.int64)
    tmp21 = tmp20.to(tl.float64)
    tmp22 = tmp17 * tmp21
    tmp23 = tmp15 + tmp22
    tmp25 = libdevice.exp(tmp24)
    tmp28 = tmp27.to(tl.int64)
    tmp29 = tmp28.to(tl.float64)
    tmp30 = tmp25 * tmp29
    tmp31 = tmp23 + tmp30
    tmp32 = tmp0 * tmp31
    tmp34 = tmp33 * tmp6
    tmp36 = tmp35 * tmp13
    tmp37 = tmp34 + tmp36
    tmp39 = tmp38 * tmp21
    tmp40 = tmp37 + tmp39
    tmp42 = tmp41 * tmp29
    tmp43 = tmp40 + tmp42
    tmp44 = tmp32 + tmp43
    tl.store(in_out_ptr0 + (x0), tmp44, xmask)
