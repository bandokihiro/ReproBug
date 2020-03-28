//
// Created by kihiro on 1/22/20.
//

#ifndef DG_IDS_H
#define DG_IDS_H

enum TaskIDs {
    TOP_LEVEL_TASK_ID,

    // mesh related tasks
    CHECK_PARTITION_TASK_ID,
    CHECK_PARTITION2_TASK_ID,
    STORE_MESHPART_NELEMPERPART_TASK_ID,
    STORE_ELEM_VOLUME_TASK_ID,
    STORE_IFACE_AREA_TASK_ID,
    STORE_BFACE_AREA_TASK_ID,
    STORE_ELEM_SURF_AREA_TASK_ID,
    STORE_ELEM_LENGTH_TASK_ID,

    // solution related task
    INIT_SOLUTION_TASK_ID,
    PRECOMPUTE_IMM_TASK_ID,
    PRINT_SOLUTION_FIELD_TASK_ID,
    ZERO_FIELD_TASK_ID,

    // writer related task
    CREATE_SOLUTION_FILE_TASK_ID,
    DUMP_SOLUTION_TASK_ID,
    CREATE_RESTART_FILE_TASK_ID,

    // residual related
    COMPUTE_VOL_RESIDUAL_TASK_ID,
    COMPUTE_IFACE_RESIDUAL_TASK_ID,
    COMPUTE_BFACE_RESIDUAL_TASK_ID,
    MULTIPLY_RESIDUAL_BY_IMM_TASK_ID,

    // stepper
    FORWARDEULER_ADVANCE_TASK_ID,
    SSPRK3_STAGE1_TASK_ID,
    SSPRK3_STAGE2_TASK_ID,
    SSPRK3_STAGE3_TASK_ID,
    RK4_STAGE1_TASK_ID,
    RK4_STAGE2_TASK_ID,
    RK4_STAGE3_TASK_ID,
    RK4_STAGE4_TASK_ID,

    // auxiliary tasks
    COMPUTE_ERROR_TASK_ID,
    COMPUTE_RESIDUAL_NORM_TASK_ID,
    COMPUTE_MAX_CFL_TASK_ID,
    COMPUTE_DIAGNOSIS_TASK_ID,
};

#endif //DG_IDS_H
