void histogram_cpu(int* data, int* histogram, int len) {
    for (int i = 0; i < len; i++) {
        int val = data[i];
        histogram[val]++;
    }
}