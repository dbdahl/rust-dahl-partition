#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

void dahl_partition__distribution__crp__sample(int n_samples, int n_items, double mass, int *ptr);

double dahl_partition__summary__binder(int n_items, const int *partition_ptr, double *epam_ptr);

void dahl_partition__summary__psm(int n_samples,
                                  int n_items,
                                  int parallel,
                                  const int *partitions_ptr,
                                  double *counts_ptr);
