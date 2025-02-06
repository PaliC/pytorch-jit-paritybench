
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 8192},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_index_add_convolution_native_group_norm_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__unsafe_index_add_convolution_native_group_norm_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 4)
    x8 = xindex // 4
    x1 = ((xindex // 4) % 64)
    tmp14 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    x6 = xindex
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex // 1024
        r4 = ((rindex // 32) % 32)
        r3 = (rindex % 32)
        r7 = rindex
        tmp0 = tl.load(in_ptr0 + (r5 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (r4), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr0 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr3 + (r7 + 8192*x6), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 16, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tmp6 = tmp5 + tmp1
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tmp10 = tmp9 + tmp1
        tmp11 = tmp9 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp9)
        tmp13 = tl.load(in_ptr1 + (tmp12 + 16*tmp8 + 256*tmp4 + 4096*x8), rmask & xmask, eviction_policy='evict_last')
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_reduce(
            tmp18, tmp19_mean, tmp19_m2, tmp19_weight, roffset == 0
        )
        tmp19_mean = tl.where(rmask & xmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(rmask & xmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(rmask & xmask, tmp19_weight_next, tmp19_weight)
        tl.store(out_ptr0 + (r7 + 8192*x6), tmp17, rmask & xmask)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tl.store(out_ptr1 + (x6), tmp19, xmask)
    tl.store(out_ptr2 + (x6), tmp20, xmask)
    tl.store(out_ptr3 + (x6), tmp21, xmask)
