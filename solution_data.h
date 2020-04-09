//
// Created by kihiro on 1/28/20.
//

#ifndef DG_SOLUTION_DATA_H
#define DG_SOLUTION_DATA_H

#include "legion.h"
#include "mesh_data.h"

struct Args {
    int iteration;
    int nIter;
};

/*! \brief Class to hold solution related regions
 *
 */
class SolutionData : public LegionData {
  public:
    /*! \brief Solution regions' fields
     *
     */
    enum FieldIDs {
        FID_SOL_RESIDUAL, //!< storage for residual
        FID_SOL_REFERENCE,
    };

    /*! \brief Pre-register all solution related tasks
     *
     */
    static void register_tasks();

    /*! \brief Constructor
     *
     * @param ctx Legion's context
     * @param runtime Legion's runtime
     * @param task_wait_all_results
     */
    SolutionData(Legion::Context ctx, Legion::HighLevelRuntime *runtime, Legion::Logger &logger_);

    /*! \brief Clean up Legion's ressources used for solution related regions
     *
     */
    void clean_up();

    struct TagLeftRightElementArgs {
        TagLeftRightElementArgs(Legion::LogicalPartition priv, Legion::LogicalPartition shared)
                : priv_elem_lp(priv), shared_elem_lp(shared) {}

        Legion::LogicalPartition priv_elem_lp;
        Legion::LogicalPartition shared_elem_lp;
    };

    /*! \brief Create solutions regions
     *
     * The fields are not initialized yet after this method is called.
     *
     * @param scheme_info
     * @param mesh_data
     */
    void create_solution_region(const MeshData &mesh_data);

    void zero_field();

    void compute_iface_residual(const int nIter, const MeshData &mesh_data);

    void copy_to_reference();

    void check(const int iteration, const int nIter);

    rtype compute_error() const;

    Legion::LogicalRegion elem_lr; //!< element logical region
    Legion::LogicalPartition elem_lp; //!< element logical partition
    Legion::LogicalPartition elem_with_halo_lp; //!< element logical partition with halo
    Legion::Domain domain; //!< partition index domain

    Legion::LogicalPartition priv_elem_lp; //!< private element logical partition
    Legion::LogicalPartition shared_elem_lp; //!< shared element logical partition
    Legion::LogicalPartition ghost_elem_lp; //!< ghost element logical partition
};

#endif //DG_SOLUTION_DATA_H
