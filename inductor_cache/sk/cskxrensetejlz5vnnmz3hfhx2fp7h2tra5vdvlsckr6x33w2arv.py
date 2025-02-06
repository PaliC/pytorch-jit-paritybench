
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_native_group_norm_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_native_group_norm_15(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 256
        tmp0 = tl.load(in_out_ptr0 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r3 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
        tl.store(in_out_ptr0 + (r5 + 2048*x4), tmp6, rmask & xmask)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tmp11 = 2048.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp15, xmask)
