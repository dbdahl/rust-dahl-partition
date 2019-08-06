#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

void dahl_partition__distribution__crp__sample(int n_partitions,
                                               int n_items,
                                               double mass,
                                               int *ptr);

void dahl_partition__summary__expected_loss(int n_partitions,
                                            int n_items,
                                            int *partition_ptr,
                                            double *psm_ptr,
                                            int loss,
                                            double *results_ptr);

void dahl_partition__summary__psm(int n_partitions,
                                  int n_items,
                                  int parallel,
                                  int *partitions_ptr,
                                  double *psm_ptr);

double dahl_partition__utils__lbell(int n);
