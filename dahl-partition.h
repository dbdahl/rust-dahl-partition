#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

void DahlPartition_Distribution_Crp_sample(int n_samples, int n_items, double mass, int *ptr);

void DahlPartition_Summary_expected_pairwise_allocation_matrix(int n_samples,
                                                               int n_items,
                                                               const int *partitions_ptr,
                                                               double *counts_ptr);
