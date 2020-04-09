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

void tag_left_right_element_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime *runtime) {
    // get private and shared logical partitions
    const typename SolutionData::TagLeftRightElementArgs *args =
            (const typename SolutionData::TagLeftRightElementArgs*)task->args;
    const LogicalPartition priv_elem_lp = args->priv_elem_lp;
    const LogicalPartition shared_elem_lp = args->shared_elem_lp;
    const LogicalPartition ghost_elem_lp = args->ghost_elem_lp;
    // get the current sub-region for both partitions
    const LogicalRegion
            priv_elem_lsr(runtime->get_logical_subregion_by_color(priv_elem_lp, task->index_point));
    const LogicalRegion
            shared_elem_lsr(runtime->get_logical_subregion_by_color(shared_elem_lp, task->index_point));
    const LogicalRegion
            ghost_elem_lsr(runtime->get_logical_subregion_by_color(ghost_elem_lp, task->index_point));

    // accessors for left and right element ID
    AffAccROPoint1 acc_elem_ID[2];
    acc_elem_ID[0] = AffAccROPoint1(regions[0], MeshData::FID_MESH_IFACE_ELEMLID);
    acc_elem_ID[1] = AffAccROPoint1(regions[0], MeshData::FID_MESH_IFACE_ELEMRID);
    // accessors for left and right element type
    AffAccWDElemType acc_elem_type[2];
    acc_elem_type[0] = AffAccWDElemType(regions[1], MeshData::FID_MESH_IFACE_ELEMLTYPE);
    acc_elem_type[1] = AffAccWDElemType(regions[1], MeshData::FID_MESH_IFACE_ELEMRTYPE);

    Domain domain = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    for (Domain::DomainPointIterator itr(domain); itr; itr++) {
        const Point<1> elemL = acc_elem_ID[0][*itr];
        const Point<1> elemR = acc_elem_ID[1][*itr];

        if (!runtime->safe_cast(ctx, elemL, priv_elem_lsr).is_null()) {
            acc_elem_type[0][*itr] = ElemType::Private;
        }
        else {
            assert(!runtime->safe_cast(ctx, elemL, shared_elem_lsr).is_null());
            acc_elem_type[0][*itr] = ElemType::Shared;
        }

        if (!runtime->safe_cast(ctx, elemR, priv_elem_lsr).is_null()) {
            acc_elem_type[1][*itr] = ElemType::Private;
        }
        else if (!runtime->safe_cast(ctx, elemR, shared_elem_lsr).is_null()) {
            acc_elem_type[1][*itr] = ElemType::Shared;
        }
        else {
            assert(!runtime->safe_cast(ctx, elemR, ghost_elem_lsr).is_null());
            acc_elem_type[1][*itr] = ElemType::Ghost;
        }
    }
}

void zero_field_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime) {
    AffAccWDrtype acc(regions[0], SolutionData::FID_SOL_RESIDUAL, N_REDOP*sizeof(rtype));
    AffAccWDrtype acc_ref(regions[0], SolutionData::FID_SOL_REFERENCE, N_REDOP*sizeof(rtype));
    Domain domain = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    for (Domain::DomainPointIterator itr(domain); itr; itr++) {
        rtype *ptr = acc.ptr(itr.p);
        rtype *ptr_ref = acc_ref.ptr(itr.p);
        for (int i=0; i<N_REDOP; i++) {
            ptr[i] = 0.;
            ptr_ref[i] = 0.;
        }
    }
}

rtype compute_error_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime) {
    AffAccROrtype acc(regions[0], SolutionData::FID_SOL_RESIDUAL, N_REDOP*sizeof(rtype));
    Domain domain = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    rtype result = 0.;
    for (Domain::DomainPointIterator itr(domain); itr; itr++) {
        const rtype *ptr = acc.ptr(itr.p);
        for (int i=0; i<N_REDOP; i++) result += ptr[i];
    }
    return result;
}

void update_iface_residual(const int nCoeff, const Point<1> elem, ElemType type,
                           const vector<rtype> &res, AffAccRWrtype priv, AccReductionSum shared,
                           AccReductionSum ghost) {
    switch(type) {
        case ElemType::Private: {
            rtype *res_field = priv.ptr(elem);
            for (int i=0; i<N_REDOP; i++) res_field[i] += res[i];
            break;
        }
        case ElemType::Shared: {
            ReductionSum<N_REDOP>::LHS *lhs = shared.ptr(elem);
            ReductionSum<N_REDOP>::RHS rhs(res);
            ReductionSum<N_REDOP>::apply<true>(*lhs, rhs);
            break;
        }
        case ElemType::Ghost: {
            ReductionSum<N_REDOP>::LHS *lhs = ghost.ptr(elem);
            ReductionSum<N_REDOP>::RHS rhs(res);
            ReductionSum<N_REDOP>::apply<true>(*lhs, rhs);
            break;
        }
        default:
            assert(false);
    }
}

void compute_iface_residual_task(const Task *task,  const vector<PhysicalRegion> &regions,
        Context ctx, Runtime *runtime) {
    int nIter = *(const int *)task->args;

    // left and right element IDs
    AffAccROPoint1 acc_face_elemID[2];
    acc_face_elemID[0] = AffAccROPoint1(regions[0], MeshData::FID_MESH_IFACE_ELEMLID, sizeof(Point<1>));
    acc_face_elemID[1] = AffAccROPoint1(regions[0], MeshData::FID_MESH_IFACE_ELEMRID, sizeof(Point<1>));
    // accessors for left and right element type
    AffAccROElemType acc_elem_type[2];
    acc_elem_type[0] = AffAccROElemType(regions[0], MeshData::FID_MESH_IFACE_ELEMLTYPE, sizeof(ElemType));
    acc_elem_type[1] = AffAccROElemType(regions[0], MeshData::FID_MESH_IFACE_ELEMRTYPE, sizeof(ElemType));
    // accessor for the residual, private
    AffAccRWrtype acc_priv_res(regions[1], SolutionData::FID_SOL_RESIDUAL, N_REDOP*sizeof(rtype));
    // accessor for the residual, shared
    AccReductionSum acc_shared_res(regions[2], SolutionData::FID_SOL_RESIDUAL, 1);
    // accessor for the residual, ghost
    AccReductionSum acc_ghost_res(regions[3], SolutionData::FID_SOL_RESIDUAL, 1);

    Domain domain = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    for (Domain::DomainPointIterator itr(domain); itr; itr++) {
        Point<1> elemL = acc_face_elemID[0][*itr];
        Point<1> elemR = acc_face_elemID[1][*itr];

        vector<rtype> tmp(N_REDOP, 0.);
        int i0 = (int) itr.p[0];
        for (int k=0; k<N_REDOP; k++) tmp[k] = (rtype) (i0+k) / (rtype) (i0+1) / (rtype) nIter;

        update_iface_residual(N_REDOP, elemL, acc_elem_type[0][*itr], tmp,
                              acc_priv_res, acc_shared_res, acc_ghost_res);
        update_iface_residual(N_REDOP, elemR, acc_elem_type[1][*itr], tmp,
                              acc_priv_res, acc_shared_res, acc_ghost_res);
    }
}

void copy_to_reference_task(const Task *task,  const vector<PhysicalRegion> &regions,
        Context ctx, Runtime *runtime) {
    AffAccROrtype acc(regions[0], SolutionData::FID_SOL_RESIDUAL, N_REDOP*sizeof(rtype));
    AffAccRWrtype acc_ref(regions[1], SolutionData::FID_SOL_REFERENCE, N_REDOP*sizeof(rtype));
    Domain domain = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

    for (Domain::DomainPointIterator itr(domain); itr; itr++) {
        const rtype *ptr = acc.ptr(itr.p);
        rtype *ptr_ref = acc_ref.ptr(itr.p);
        for (int i=0; i<N_REDOP; i++) {
            ptr_ref[i] = ptr[i];
        }
    }
}

void check_task(const Task *task,  const vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime) {
    Args arg = *(const Args *)task->args;
    if (arg.iteration>0) {
        AffAccROrtype acc_res(regions[0], SolutionData::FID_SOL_RESIDUAL, N_REDOP*sizeof(rtype));
        AffAccROrtype acc_ref(regions[0], SolutionData::FID_SOL_REFERENCE, N_REDOP*sizeof(rtype));
        Domain domain = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
        for (Domain::DomainPointIterator itr(domain); itr; itr++) {
            for (int i=0; i<N_REDOP; i++) {
                const rtype *ptr = acc_res.ptr(itr.p);
                const rtype *ptr_ref = acc_ref.ptr(itr.p);
                const rtype ref_value = (arg.iteration+1)*ptr_ref[i];
                rtype err = fabs(ptr[i] - ref_value) / ref_value;
                assert(err < 1e-12);
            }
        }
        if (task->index_point == Point<1>(0)) {
            cout << "Checked result of iteration " << arg.iteration << endl;
        }
    }
}

void SolutionData::register_tasks() {
    {
        TaskVariantRegistrar registrar(TAG_LEFT_RIGHT_ELEMENT_TASK_ID, "tag_left_right_element_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<tag_left_right_element_task> (registrar, "tag_left_right_element_task");
    }
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
    {
        TaskVariantRegistrar registrar(COMPUTE_IFACE_RESIDUAL_TASK_ID,
            "compute_iface_residual_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<compute_iface_residual_task> (registrar,
            "compute_iface_residual_task");
    }
    {
        TaskVariantRegistrar registrar(COPY_TO_REFERENCE_TASK_ID, "copy_to_reference_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<copy_to_reference_task> (registrar, "copy_to_reference_task");
    }
    {
        TaskVariantRegistrar registrar(CHECK_TASK_ID, "check_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<check_task> (registrar, "check_task");
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
    allocator.allocate_field(N_REDOP*sizeof(rtype), FID_SOL_REFERENCE);

    runtime->attach_name(fs, FID_SOL_RESIDUAL, "sol_residual");
    runtime->attach_name(fs, FID_SOL_REFERENCE, "sol_reference");

    elem_lr = runtime->create_logical_region(ctx, mesh_data.elem_lr.get_index_space(), fs);
    runtime->attach_name(elem_lr, "sol_elem_logical_region");

    elem_lp = runtime->get_logical_partition(ctx, elem_lr, mesh_data.elem_lp.get_index_partition());
    runtime->attach_name(elem_lp, "sol_elem_logical_partition");

    elem_with_halo_lp = runtime->get_logical_partition(ctx, elem_lr,
        mesh_data.elem_with_halo_lp.get_index_partition());
    runtime->attach_name(elem_with_halo_lp, "sol_elem_with_halo_logical_partition");

    // create logical partitions for private, shared and ghost elements
    priv_elem_lp = runtime->get_logical_partition(ctx, elem_lr, mesh_data.priv_elem_ip);
    runtime->attach_name(priv_elem_lp, "solution_private_logical_partition");
    shared_elem_lp = runtime->get_logical_partition(ctx, elem_lr, mesh_data.shared_elem_ip);
    runtime->attach_name(shared_elem_lp, "solution_shared_logical_partition");
    ghost_elem_lp = runtime->get_logical_partition(ctx, elem_lr, mesh_data.ghost_elem_ip);
    runtime->attach_name(ghost_elem_lp, "solution_ghost_logical_partition");

    domain = Domain::from_rect<1>(Arrays::Rect<1>(Arrays::Point<1>(0),
            Arrays::Point<1>(mesh_data.nPart-1)));

    TagLeftRightElementArgs args(priv_elem_lp, shared_elem_lp, ghost_elem_lp);
    IndexLauncher index_launcher(TAG_LEFT_RIGHT_ELEMENT_TASK_ID, domain,
            TaskArgument(&args, sizeof(TagLeftRightElementArgs)), ArgumentMap());
    // left and right element ID
    RegionRequirement req(mesh_data.iface_lp, 0, READ_ONLY, EXCLUSIVE, mesh_data.iface_lr);
    req.add_field(MeshData::FID_MESH_IFACE_ELEMLID);
    req.add_field(MeshData::FID_MESH_IFACE_ELEMRID);
    index_launcher.add_region_requirement(req);
    // left and right element type
    req = RegionRequirement(mesh_data.iface_lp, 0, WRITE_DISCARD, EXCLUSIVE, mesh_data.iface_lr);
    req.add_field(MeshData::FID_MESH_IFACE_ELEMLTYPE);
    req.add_field(MeshData::FID_MESH_IFACE_ELEMRTYPE);
    index_launcher.add_region_requirement(req);
    // run
    runtime->execute_index_space(ctx, index_launcher);
}

void SolutionData::zero_field() {
    IndexLauncher index_launcher(ZERO_FIELD_TASK_ID, domain, TaskArgument(), ArgumentMap());
    // solution region
    RegionRequirement req(elem_lp, 0, WRITE_DISCARD, EXCLUSIVE, elem_lr);
    req.add_field(FID_SOL_RESIDUAL);
    req.add_field(FID_SOL_REFERENCE);
    index_launcher.add_region_requirement(req);
    // run
    runtime->execute_index_space(ctx, index_launcher);
}

void SolutionData::compute_iface_residual(const int nIter, const MeshData &mesh_data) {
    IndexLauncher index_launcher(COMPUTE_IFACE_RESIDUAL_TASK_ID, domain,
                                 TaskArgument(&nIter, sizeof(int)), ArgumentMap());
    // mesh region: iface data
    RegionRequirement req(mesh_data.iface_lp, 0, READ_ONLY, EXCLUSIVE, mesh_data.iface_lr);
    vector<FieldID> fields{ MeshData::FID_MESH_IFACE_ELEMLID,
                            MeshData::FID_MESH_IFACE_ELEMRID,
                            MeshData::FID_MESH_IFACE_ELEMLTYPE,
                            MeshData::FID_MESH_IFACE_ELEMRTYPE, };
    req.add_fields(fields);
    index_launcher.add_region_requirement(req);
    // solution region: residual, private
    req = RegionRequirement(priv_elem_lp, 0, READ_WRITE, EXCLUSIVE, elem_lr);
    req.add_field(FID_SOL_RESIDUAL);
    index_launcher.add_region_requirement(req);
    // solution region: residual, shared
    req = RegionRequirement(shared_elem_lp, 0, 1, EXCLUSIVE, elem_lr);
    req.add_field(FID_SOL_RESIDUAL);
    index_launcher.add_region_requirement(req);
    // solution region: residual, ghost
    req = RegionRequirement(ghost_elem_lp, 0, 1, EXCLUSIVE, elem_lr);
    req.add_field(FID_SOL_RESIDUAL);
    index_launcher.add_region_requirement(req);
    // run
    runtime->execute_index_space(ctx, index_launcher);
}

void SolutionData::copy_to_reference() {
    IndexLauncher index_launcher(COPY_TO_REFERENCE_TASK_ID, domain, TaskArgument(), ArgumentMap());

    RegionRequirement req(elem_lp, 0, READ_ONLY, EXCLUSIVE, elem_lr);
    req.add_field(FID_SOL_RESIDUAL);
    index_launcher.add_region_requirement(req);

    req = RegionRequirement(elem_lp, 0, READ_WRITE, EXCLUSIVE, elem_lr);
    req.add_field(FID_SOL_REFERENCE);
    index_launcher.add_region_requirement(req);

    runtime->execute_index_space(ctx, index_launcher);
}

void SolutionData::check(const int iteration, const int nIter) {
    Args arg;
    arg.iteration = iteration;
    arg.nIter = nIter;
    IndexLauncher index_launcher(CHECK_TASK_ID, domain, TaskArgument(&arg, sizeof(Args)), ArgumentMap());
    RegionRequirement req(elem_lp, 0, READ_ONLY, EXCLUSIVE, elem_lr);
    req.add_field(SolutionData::FID_SOL_RESIDUAL);
    req.add_field(SolutionData::FID_SOL_REFERENCE);
    index_launcher.add_region_requirement(req);
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
    return f.get_result<rtype>(true);
}