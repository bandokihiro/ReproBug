//
// Created by kihiro on 1/28/20.
//

#ifndef DG_MESH_DATA_H
#define DG_MESH_DATA_H

#include "legion.h"
#include "mesh.h"

enum class ElemType {
    Private,
    Shared,
    Ghost,
};

class LegionData {
public:
    /*! \brief Constructor
     *
     * @param ctx_
     * @param runtime_
     * @param task_wait_all_results_
     */
    LegionData(Legion::Context ctx_, Legion::HighLevelRuntime *runtime_, Legion::Logger &logger_)
            : ctx(ctx_), runtime(runtime_),  logger(logger_) {}

    Legion::Context ctx; //!< Legion's context
    Legion::HighLevelRuntime *runtime; //!< Legion's runtime
    Legion::Logger logger;
};

/*! \brief Class holding the mesh related regions
 */
class MeshData : public LegionData {
  public:
    /*! \brief Mesh regions' fields
     *
     * Offset of 100 in order to avod enum collisions with fields from other regions.
     */
    enum FieldIDs {
        FID_MESH_ELEM_PARTID, //!< element partition ID
        FID_MESH_ELEM_GHOST_BITMASK, //!< bitmask discribing the subregions that are adjacent to the element
        FID_MESH_IFACE_ELEMLID, //!< interior face's left element
        FID_MESH_IFACE_ELEMRID, //!< interior face's right element

        FID_MESH_IFACE_ELEMLTYPE, //!< type of the left element
        FID_MESH_IFACE_ELEMRTYPE, //!< type of the left element
    };

    /*! \brief Constructor
     *
     * @param ctx Legion's context
     * @param runtime Legion's runtime
     * @param task_wait_all_results
     */
    MeshData(Legion::Context ctx, Legion::HighLevelRuntime *runtime, Legion::Logger &logger);

    /*! \brief Clean up Legion's ressources used for mesh related regions
     *
     */
    void clean_up();

    /*! \brief Initialize the mesh regions
     *
     * Create mesh regions from the mesh object. This object is not needed anymore after these
     * regions have been initialized.
     *
     * @param mesh mesh object
     */
    void init_mesh_region(const Mesh &mesh);

    /*! Partition the mesh regions
     *
     * Generate 2 partitions for the elements (one without and with halo elements).
     *
     * @param nPart number of partitions
     */
    void partition_mesh_region();

    void reinit_mesh_region(const Mesh &mesh);

    /*! Check partitioning
     *
     * Print the partition data. Only used for verification purposes. Use it on a small mesh and in
     * serial.
     */
    void check_partitioning();

    int nElem; //!< number of elements
    int nPart; //!< number of partitions
    int nIface;

    Legion::LogicalRegion elem_lr; //!< element logical region
    Legion::LogicalPartition elem_lp; //!< element logical partition without halo elements
    Legion::LogicalPartition elem_with_halo_lp; //!< element logical partition with halo elements
    Legion::LogicalRegion iface_lr; //!< interior face logical region
    Legion::LogicalPartition iface_lp; //!< interior face logical partition
    Legion::LogicalPartition iface_all_lp; //!< all interior face logical partition

    Legion::IndexSpace all_shared_is; //!< index space for all elements that are shared
    Legion::IndexPartition priv_elem_ip; //!< index partition for private elements
    Legion::IndexPartition shared_elem_ip; //!< index partition for shared elements
    Legion::IndexPartition ghost_elem_ip; //!< index partition for shared elements

  private:
    /*! \brief Initialize the mesh element region
     *
     * @param mesh mesh object
     */
    void init_mesh_region_elem(const Mesh &mesh);

    /*! \brief Initialize the mesh interior face region
     *
     * @param mesh mesh object
     */
    void init_mesh_region_iFace(const Mesh &mesh);

    /*! \brief Check the initial partitioning (the one without halo elements)
     *
     * Print stuff. Only used for verification purposes. Use it on a small mesh and in serial.
     */
    void check_initial_partitioning();

    /*! \brief Check the partitioning containing halo
     *
     * Print stuff. Only used for verification purposes. Use it on a small mesh and in serial.
     */
    void check_partitioning_with_halo();

    Legion::Domain domain; //!< domain associated with the partitioninig index space
};


#endif //DG_MESH_REGION_H
