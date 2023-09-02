void transpose_cpu(float* idata, float* odata, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            odata[j * h + i] = idata[i * w + j];
        }
    }
}