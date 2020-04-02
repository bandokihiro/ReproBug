//
// Created by kihiro on 3/27/20.
//

#include <string>
#include "toml11/toml.hpp"
#include "legion.h"
#include "mesh.h"
#include "mesh_data.h"
#include "solution_data.h"
#include "redop.h"
#include "ids.h"

using namespace std;
using namespace Legion;

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
    Logger logger("DG-solver");
    stringstream msg;
    auto input_info = toml::parse("input.toml");
    auto nParts = toml::find<int>(input_info, "Mesh", "npartitions");
    auto iter = toml::find<int>(input_info, "Mesh", "iter");
    auto mesh_file = toml::find<string>(input_info, "Mesh", "file");
    Mesh mesh(input_info);
    mesh.partition(nParts);
    msg.str(std::string());
    msg << mesh;
    runtime->print_once(ctx, stdout, msg.str().c_str());

    MeshData mesh_data(ctx, runtime, logger);
    mesh_data.init_mesh_region(mesh);
    mesh_data.partition_mesh_region(mesh.nPart);
    runtime->print_once(ctx, stdout, "Mesh region initialized and partitioned\n");


    SolutionData solution_data(ctx, runtime, logger);
    solution_data.create_solution_region(mesh_data);
    runtime->print_once(ctx, stdout, "Solution region created\n");
    solution_data.zero_field();

    for (int i=0; i<iter; i++) {
        solution_data.zero_field();
        rtype sum = solution_data.compute_error();
        std::cout << "iter " << i << std::endl;
    }
    rtype sum = solution_data.compute_error();
    char msg2[1000];
    sprintf(msg2, "Error = %.10e\n", sum);
    runtime->print_once(ctx, stdout, msg2);

    solution_data.clean_up();
    mesh_data.clean_up();
}

int main(int argc, char *argv[]) {
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task> (registrar, "top_level_task");
    }

    SolutionData::register_tasks();
    Runtime::register_reduction_op<ReductionSum<N_REDOP>>(1);

    return Runtime::start(argc, argv);
}
