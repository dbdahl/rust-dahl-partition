#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

extern void Rprintf(const char *fmt);

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

void dahl_partition__summary__minimize_by_enumeration(int32_t n_items,
                                                      double *psm_ptr,
                                                      int32_t loss,
                                                      int32_t *results_label_ptr);

void dahl_partition__summary__minimize_by_salso(int32_t n_items,
                                                double *psm_ptr,
                                                int32_t candidates,
                                                int32_t max_scans,
                                                int32_t loss,
                                                int32_t max_size,
                                                int32_t parallel,
                                                int32_t *results_labels_ptr,
                                                double *results_expected_loss_ptr,
                                                int32_t *results_scans_ptr);

void dahl_partition__summary__psm(int32_t n_partitions,
                                  int32_t n_items,
                                  int32_t parallel,
                                  int32_t *partitions_ptr,
                                  double *psm_ptr);

double dahl_partition__utils__bell(int32_t n);

void dahl_partition__utils__enumerate(int32_t n_partitions,
                                      int32_t n_items,
                                      int32_t *partitions_ptr);

double dahl_partition__utils__lbell(int32_t n);
