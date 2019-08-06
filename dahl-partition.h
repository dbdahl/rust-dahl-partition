#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

void dahl_partition__distribution__crp__sample(int32_t n_partitions,
                                               int32_t n_items,
                                               double mass,
                                               int32_t *ptr);

void dahl_partition__summary__expected_loss(int32_t n_partitions,
                                            int32_t n_items,
                                            int32_t *partition_ptr,
                                            double *psm_ptr,
                                            int32_t loss,
                                            double *results_ptr);

void dahl_partition__summary__psm(int32_t n_partitions,
                                  int32_t n_items,
                                  int32_t parallel,
                                  int32_t *partitions_ptr,
                                  double *psm_ptr);

double dahl_partition__utils__bell(int32_t n);

double dahl_partition__utils__lbell(int32_t n);
