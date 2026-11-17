// Isolated wider projection tile; production remains four-token until measured.
extern "C" __global__ void prefill_matmul8(const u8 *w, int kind, const float *x,
                                         float *y, int width, int rows, int tokens) {
    switch (kind) {
        case 0: prompt_tile<0, false, 8>(w, x, y, width, rows, tokens); break;
        case 1: prompt_tile<1, false, 8>(w, x, y, width, rows, tokens); break;
        case 6: prompt_tile<6, false, 8>(w, x, y, width, rows, tokens); break;
        case 8: prompt_tile<8, false, 8>(w, x, y, width, rows, tokens); break;
        case 12: prompt_tile<12, false, 8>(w, x, y, width, rows, tokens); break;
        case 14: prompt_tile<14, false, 8>(w, x, y, width, rows, tokens); break;
    }
}
