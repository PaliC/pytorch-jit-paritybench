
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_group_norm_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_group_norm_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r3 = rindex // 256
    x0 = (xindex % 32)
    r2 = (rindex % 256)
    x1 = xindex // 32
    r5 = rindex
    x4 = xindex
    tmp17 = tl.load(in_ptr3 + (r5 + 1024*x4), None)
    tmp0 = r3 + 4*x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + 256*(r3 + 4*x0) + 16384*x1), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (r2 + 256*((-64) + r3 + 4*x0) + 8192*x1), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (r2 + 256*((-96) + r3 + 4*x0) + 8192*x1), tmp11, other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tl.full([1], 1024, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp19 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = 1024.0
    tmp33 = tmp31 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tl.store(out_ptr0 + (r5 + 1024*x4), tmp16, None)
    tl.store(out_ptr3 + (x4), tmp36, None)
    tl.store(out_ptr1 + (x4), tmp26, None)
    tl.store(out_ptr2 + (x4), tmp31, None)
