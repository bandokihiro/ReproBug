//
// Created by kihiro on 1/28/20.
//

#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include "H5Cpp.h"
#include "legion.h"
#include "mesh_data.h"
#include "solution_data.h"
#include "ids.h"
#include "redop.h"
#include "typedefs.h"

using namespace H5;
using namespace Legion;
using namespace LegionRuntime;
using namespace std;

void zero_field_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime) {
    AffAccWDrtype acc(regions[0], SolutionData::FID_SOL_RESIDUAL, N_REDOP*sizeof(rtype));
}

rtype compute_error_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime) {
    AffAccROrtype acc(regions[0], SolutionData::FID_SOL_RESIDUAL, N_REDOP*sizeof(rtype));
    rtype result = 0.;
    return result;
}

void SolutionData::register_tasks() {
    {
        TaskVariantRegistrar registrar(ZERO_FIELD_TASK_ID, "zero_field_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<zero_field_task> (registrar,
            "zero_field_task");
    }
    {
        TaskVariantRegistrar registrar(COMPUTE_ERROR_TASK_ID, "compute_error");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<rtype, compute_error_task> (registrar,
            "compute_error");
    }
}

SolutionData::SolutionData(Context ctx, HighLevelRuntime *runtime, Legion::Logger &logger_) :
    LegionData(ctx, runtime, logger_) {}

void SolutionData::clean_up() {
    runtime->destroy_index_partition(ctx, elem_lp.get_index_partition());
    runtime->destroy_index_partition(ctx, elem_with_halo_lp.get_index_partition());

    runtime->destroy_logical_partition(ctx, elem_lp);
    runtime->destroy_logical_partition(ctx, elem_with_halo_lp);

    runtime->destroy_field_space(ctx, elem_lr.get_field_space());
    runtime->destroy_logical_region(ctx, elem_lr);
}

void SolutionData::create_solution_region(const MeshData &mesh_data) {
    FieldSpace fs = runtime->create_field_space(ctx);
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);

    allocator.allocate_field(N_REDOP*sizeof(rtype), FID_SOL_RESIDUAL);
    runtime->attach_name(fs, FID_SOL_RESIDUAL, "sol_residual");

    elem_lr = runtime->create_logical_region(ctx, mesh_data.elem_lr.get_index_space(), fs);
    runtime->attach_name(elem_lr, "sol_elem_logical_region");

    elem_lp = runtime->get_logical_partition(ctx, elem_lr, mesh_data.elem_lp.get_index_partition());
    runtime->attach_name(elem_lp, "sol_elem_logical_partition");

    elem_with_halo_lp = runtime->get_logical_partition(ctx, elem_lr,
        mesh_data.elem_with_halo_lp.get_index_partition());
    runtime->attach_name(elem_with_halo_lp, "sol_elem_with_halo_logical_partition");

    domain = Domain::from_rect<1>(Arrays::Rect<1>(Arrays::Point<1>(0),
        Arrays::Point<1>(mesh_data.nPart-1)));
}

void SolutionData::zero_field() {
    IndexLauncher index_launcher(ZERO_FIELD_TASK_ID, domain, TaskArgument(), ArgumentMap());
    // solution region
    RegionRequirement req(elem_lp, 0, WRITE_DISCARD, EXCLUSIVE, elem_lr);
    req.add_field(FID_SOL_RESIDUAL);
    index_launcher.add_region_requirement(req);
    // run
    runtime->execute_index_space(ctx, index_launcher);
}

rtype SolutionData::compute_error() const {
    IndexLauncher index_launcher(COMPUTE_ERROR_TASK_ID, domain, TaskArgument(), ArgumentMap());
    // solution region
    RegionRequirement req(elem_lp, 0, READ_ONLY, EXCLUSIVE, elem_lr);
    req.add_field(FID_SOL_RESIDUAL);
    index_launcher.add_region_requirement(req);
    // run
    Future f =  runtime->execute_index_space(ctx, index_launcher, SumReduction<rtype>::REDOP_ID);
    // collect result
    rtype result = 0.0;
    result = f.get_result<rtype>();
    return result;
}
