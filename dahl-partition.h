#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

void dahl_partition__distribution__crp__sample(int n_samples, int n_items, double mass, int *ptr);

void dahl_partition__summary__epam__expected_pairwise_allocation_matrix(int n_samples,
                                                                        int n_items,
                                                                        const int *partitions_ptr,
                                                                        double *counts_ptr);
