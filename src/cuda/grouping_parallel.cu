// One warp per expert; descriptors retain the serial grouping's stable order.
extern "C" __global__ void group_parallel_count32(const int *ids, int *counts, int assignments) {
    int expert = blockIdx.x, lane = threadIdx.x, count = 0;
    for (int i = lane; i < assignments; i += 32) count += ids[i] == expert;
    for (int shift = 16; shift > 0; shift /= 2)
        count += __shfl_down_sync(0xffffffff, count, shift);
    if (lane == 0) counts[expert] = count;
}

template<int Tile>
__device__ __forceinline__ void group_parallel_emit(
    const int *ids, const int *counts, int *groups, int assignments, int experts, int slots) {
    constexpr int stride = Tile + 1;
    int expert = blockIdx.x, lane = threadIdx.x, base = 0;
    for (int i = lane; i < expert; i += 32) base += (counts[i] + Tile - 1) / Tile;
    for (int shift = 16; shift > 0; shift /= 2)
        base += __shfl_down_sync(0xffffffff, base, shift);
    base = __shfl_sync(0xffffffff, base, 0);
    if (expert == experts) {
        for (int i = stride * base + lane; i < stride * slots; i += 32) groups[i] = -1;
        return;
    }

    int count = counts[expert];
    if (count == 0) return;
    int tiles = (count + Tile - 1) / Tile;
    for (int tile = lane; tile < tiles; tile += 32)
        groups[stride * (base + tile)] = expert;
    for (int i = count + lane; i < tiles * Tile; i += 32)
        groups[stride * (base + i / Tile) + 1 + i % Tile] = -1;

    int seen = 0;
    for (int offset = 0; offset < assignments && seen < count; offset += 32) {
        int i = offset + lane;
        bool match = i < assignments && ids[i] == expert;
        unsigned mask = __ballot_sync(0xffffffff, match);
        if (match) {
            int j = seen + __popc(mask & ((1u << lane) - 1));
            groups[stride * (base + j / Tile) + 1 + j % Tile] = i;
        }
        seen += __popc(mask);
    }
}

extern "C" __global__ void group_parallel_emit32(
    const int *ids, const int *counts, int *groups, int assignments, int experts, int slots) {
    group_parallel_emit<32>(ids, counts, groups, assignments, experts, slots);
}

#ifdef CUDA_M8_PROTOTYPES
extern "C" __global__ void group_parallel_emit8(
    const int *ids, const int *counts, int *groups, int assignments, int experts, int slots) {
    group_parallel_emit<8>(ids, counts, groups, assignments, experts, slots);
}

// Full tiles occupy the prefix of schedule; adjacent short ranks share a CTA.
// In a full entry the second field is -2; in a singleton short entry it is -1.
extern "C" __global__ void group_parallel_mixed_schedule(
    const int *groups, int slots, int *schedule) {
    int lane = threadIdx.x;
    constexpr unsigned Mask = 0xffffffffu;
    int full = 0;
    for (int base = 0; base < slots; base += 32) {
        int i = base + lane;
        bool live = i < slots && groups[i * 33] >= 0;
        full += __popc(__ballot_sync(Mask, live && groups[i * 33 + 9] >= 0));
    }
    int seen_full = 0, seen_short = 0;
    unsigned lower = (1u << lane) - 1;
    for (int base = 0; base < slots; base += 32) {
        int i = base + lane;
        bool live = i < slots && groups[i * 33] >= 0;
        bool wide = live && groups[i * 33 + 9] >= 0;
        unsigned wide_bits = __ballot_sync(Mask, wide);
        unsigned short_bits = __ballot_sync(Mask, live && !wide);
        if (wide) {
            int dst = 2 * (seen_full + __popc(wide_bits & lower));
            schedule[dst] = i;
            schedule[dst + 1] = -2;
        } else if (live) {
            int rank = seen_short + __popc(short_bits & lower);
            schedule[2 * full + rank] = i;
        }
        seen_full += __popc(wide_bits);
        seen_short += __popc(short_bits);
    }
    for (int i = 2 * full + seen_short + lane; i < 2 * slots; i += 32)
        schedule[i] = -1;
}

extern "C" __global__ void group_parallel_same_expert_schedule(
    const int *groups, int slots, int *schedule) {
    int lane = threadIdx.x;
    unsigned lower = (1u << lane) - 1;
    int seen = 0;
    for (int base = 0; base < slots; base += 32) {
        int i = base + lane;
        int expert = i < slots ? groups[i * 33] : -1;
        int preceding = 0;
        if (expert >= 0) {
            for (int j = i - 1; j >= 0 && groups[j * 33] == expert; j--)
                preceding++;
        }
        bool first = expert >= 0 && (preceding % 2 == 0);
        unsigned mask = __ballot_sync(0xffffffffu, first);
        if (first) {
            int entry = seen + __popc(mask & lower);
            bool paired = i + 1 < slots && groups[(i + 1) * 33] == expert;
            schedule[entry * 2] = i;
            schedule[entry * 2 + 1] = paired ? i + 1 : -2;
        }
        seen += __popc(mask);
    }
    for (int i = 2 * seen + lane; i < 2 * slots; i += 32)
        schedule[i] = -1;
}
#endif
