void reduce_cpu(int* data, int len, int* result) {
    for (int i = 0; i < len; i++) {
        result[0] += data[i];
    }
}