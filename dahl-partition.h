#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

void dahl_partition__distribution__crp__sample(int n_samples, int n_items, double mass, int *ptr);

void dahl_partition__summary__expected_loss(int n_samples,
                                            int n_items,
                                            const int *partition_ptr,
                                            double *psm_ptr,
                                            int loss,
                                            double *results_ptr);

void dahl_partition__summary__psm(int n_samples,
                                  int n_items,
                                  int parallel,
                                  const int *partitions_ptr,
                                  double *psm_ptr);
