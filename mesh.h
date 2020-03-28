//
// Created by kihiro on 1/23/20.
//

#ifndef DG_MESH_H
#define DG_MESH_H

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "H5Cpp.h"
#include "metis.h"
#include "toml11/toml.hpp"
#include "types.h"

/*! \brief Mesh class
 *
 */
class Mesh {
  public:
    /*! \brief Default constructor
     *
     */
    Mesh() = default;

    /*! \brief Constructor from input file information
     *
     * @param input_info
     */
    explicit Mesh(const toml::value &input_info);

    /*! \brief Read HDF5 mesh file
     *
     * @param mesh_file_name
     *
     * Currently allowed here for testing.
     */
    void read_mesh(const std::string &mesh_file_name); // TODO: hide this function

    /*! \brief Partition the mesh sequentially using metis
     *
     * @param nparts
     */
    void partition(int nparts);

    /*! \brief << operator
     *
     * @param os
     * @param mesh
     * @return
     */
    friend std::ostream& operator<<(std::ostream &os, const Mesh &mesh);

    int nElem; //!< number of elements
    int nNode; //!< number of nodes
    int nNode_per_elem; //!< number of nodes per element
    int nIface; //!< number of interior faces
    int nPart; //!< number of partitions requested
    int dim; //!< number of spatial dimentions
    int nBFG; //!< number of boundary face groups
    int nBFace; //!< total number of boundary faces
    int order; //!< geometric order
    std::vector<std::string> BFG_names; //!< name of boundary groups
    std::map<std::string, int> BFG_to_nBFace; //!< map from BFG name to number of BFace in that group
    /*! \brief Map from BFG name to boundary data
     *
     * Each element of the vector contains the following information;
     * - the element ID
     * - the face ID from the point of view of the element
     * - the face orientation from the point of view of the element
     */
    std::map<std::string, std::vector<std::vector<int>>> BFG_to_data;
    std::vector<std::vector<rtype>> coord; //!< vector containing node coordinates for each node
    /*! \brief vector relating faces to adjacent elements
     *
     * Each element of the vector contains the following information
     * (first for left then for right element):
     * - element ID
     * - face ID from the point of view of the element
     * - face orientation from the point of view of the element
     */
    std::vector<std::vector<int>> IFace_to_elem;
    std::vector<int> elem_num_IFace;
    std::vector<std::vector<int>> elem_to_IFace;
    std::vector<int> elem_num_BFace;
    std::vector<std::vector<int>> elem_to_BFace;
    std::vector<idx_t> eptr; //!< for metis
    std::vector<idx_t> eind; //!< for metis
    std::vector<idx_t> elem_part_id; //!< vector of partition ID for each element
    std::vector<idx_t> node_part_id; //!< vector of partition ID for each node

  private:
    bool partitioned; //!< boolean indicating whether the mesh is partitioned
    void read_boundary_faces(H5::H5File &file); //!< read boundary faces when they exist
};

#endif //DG_MESH_H
