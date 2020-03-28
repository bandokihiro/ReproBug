//
// Created by kihiro on 1/23/20.
//

#include <algorithm>
#include <iostream>
#include <string>
#include "H5Cpp.h"
#include "metis.h"
#include "toml11/toml.hpp"
#include "mesh.h"

using namespace std;
using namespace H5;

// identifiers from Eric's pre-processing tool
const string DSET_NELEM("nElem");
const string DSET_NNODE("nNode");
const string DSET_NIFACE("nIFace");
const string DSET_NNODE_PER_ELEM("nNodePerElem");
const string DSET_NODE_COORD("NodeCoords");
const string DSET_DIMS("Dimension");
const string DSET_ELEM_TO_NODES("Elem2Nodes");
const string DSET_IFACE("IFaceData");
const string DSET_QORDER("QOrder");

Mesh::Mesh(const toml::value &input_info) : partitioned(false) {
    string mesh_file_name = toml::find<string>(input_info, "Mesh", "file");
    if (input_info.contains("Boundaries")) {
        BFG_names = toml::find<vector<string>>(input_info, "Boundaries", "names");
    }
    else {
        BFG_names.resize(0);
        nBFG = 0;
        nBFace = 0;
    }
    read_mesh(mesh_file_name);
}

void Mesh::read_mesh(const string &mesh_file_name) {
    try {
        // turn off the auto-printing when failure occurs
        hsize_t dims[2]; // buffer to store size in each dimensions

        // open hdf5 file
        H5File file(mesh_file_name, H5F_ACC_RDONLY);

        // fetch all scalars first
        dims[0] = 1;
        int rank = 1;
        DataSpace mspace(rank, dims);

        // number of spatial dimensions
        DataSet dataset   = file.openDataSet(DSET_DIMS);
        DataSpace dataspace = dataset.getSpace();
        dataset.read(&dim, PredType::NATIVE_INT, mspace, dataspace);
        // number of elements
        dataset   = file.openDataSet(DSET_NELEM);
        dataspace = dataset.getSpace();
        dataset.read(&nElem, PredType::NATIVE_INT, mspace, dataspace);
        // number of nodes
        dataset   = file.openDataSet(DSET_NNODE);
        dataspace = dataset.getSpace();
        dataset.read(&nNode, PredType::NATIVE_INT, mspace, dataspace);
        // number of nodes per element
        dataset   = file.openDataSet(DSET_NNODE_PER_ELEM);
        dataspace = dataset.getSpace();
        dataset.read(&nNode_per_elem, PredType::NATIVE_INT, mspace, dataspace);
        // number of interior faces
        dataset   = file.openDataSet(DSET_NIFACE);
        dataspace = dataset.getSpace();
        dataset.read(&nIface, PredType::NATIVE_INT, mspace, dataspace);
        // geometric order of the mesh
        dataset   = file.openDataSet(DSET_QORDER);
        dataspace = dataset.getSpace();
        dataset.read(&order, PredType::NATIVE_INT, mspace, dataspace);

        // resize internal structures
        eptr.resize(nElem + 1);
        eind.resize(nElem * nNode_per_elem);
        coord.resize(nNode);
        IFace_to_elem.resize(nIface);

        elem_num_IFace.resize(nElem);
        elem_to_IFace.resize(nElem);
        elem_num_BFace.resize(nElem);
        elem_to_BFace.resize(nElem);
        for (int i = 0; i < nElem; i++){
            elem_num_IFace[i] = 0.0;
            elem_to_IFace[i].resize(6); // max number of faces
            elem_num_BFace[i] = 0.0;
            elem_to_BFace[i].resize(6); // max number of faces
        }

        // fetch elemID->nodeID
        dims[0] = nElem;
        dims[1] = nNode_per_elem;
        rank = 2;
        mspace = DataSpace(rank, dims);
        dataset   = file.openDataSet(DSET_ELEM_TO_NODES);
        dataspace = dataset.getSpace();
        // ordering: elemID X nodeID (last is fastest)
        dataset.read(eind.data(), PredType::NATIVE_INT, mspace, dataspace);

        // fetch node coordinates
        vector<rtype> buff(dim * nNode, 0.);
        dims[0] = nNode;
        dims[1] = dim;
        rank = 2;
        mspace = DataSpace(rank, dims);
        dataset = file.openDataSet(DSET_NODE_COORD);
        dataspace = dataset.getSpace();
        // ordering: nodeID X coordinates (last is fastest)
        dataset.read(buff.data(), PredType::NATIVE_DOUBLE, mspace, dataspace);
        // fill coordinates
        for (int iNode = 0; iNode<nNode; iNode++) {
            coord[iNode].resize(dim);
            for (int idim=0; idim<dim; idim++) {
                coord[iNode][idim] = buff[iNode*dim + idim];
            }
        }

        // fetch IFace->elem and IFace->node
        vector<int> buff_int;
        buff_int.resize(nIface * 6);
        dims[0] = nIface;
        dims[1] = 6;
        rank = 2;
        mspace = DataSpace(rank, dims);
        dataset = file.openDataSet(DSET_IFACE);
        dataspace = dataset.getSpace();
        dataset.read(buff_int.data(), PredType::NATIVE_INT, mspace, dataspace);
        for (int i=0; i<nIface; i++) {
            IFace_to_elem[i].resize(6);
            IFace_to_elem[i][0] = buff_int[6*i + 0]; // left element ID
            IFace_to_elem[i][1] = buff_int[6*i + 1]; // face ID for the left element
            IFace_to_elem[i][2] = buff_int[6*i + 2]; // orientation for the left element
            IFace_to_elem[i][3] = buff_int[6*i + 3]; // right element ID
            IFace_to_elem[i][4] = buff_int[6*i + 4]; // face ID for the right element
            IFace_to_elem[i][5] = buff_int[6*i + 5]; // orientation for the right element

            // Set reverse mapping
            elem_to_IFace[buff_int[6*i + 0]][elem_num_IFace[buff_int[6*i + 0]]] = i;
            elem_to_IFace[buff_int[6*i + 3]][elem_num_IFace[buff_int[6*i + 3]]] = i;
            elem_num_IFace[buff_int[6*i + 0]]++;
            elem_num_IFace[buff_int[6*i + 3]]++;
        }

        // fill eptr that indicates where data for node i in eind is
        idx_t counter = 0;
        for (int i = 0; i < nElem + 1; i++) {
            eptr[i] = counter;
            counter += nNode_per_elem;
        }

        // resize partition ID container
        elem_part_id.resize(nElem);
        node_part_id.resize(nNode);

        // read boundary faces
        if (BFG_names.size()>0) {
            read_boundary_faces(file);
        }
        else {
            nBFG = 0;
            nBFace = 0;
        }
    }

        // catch failure caused by the H5File operations
    catch( FileIException error ) {
        error.printErrorStack();
    }

        // catch failure caused by the DataSet operations
    catch( DataSetIException error ) {
        error.printErrorStack();
    }

        // catch failure caused by the DataSpace operations
    catch( DataSpaceIException error ) {
        error.printErrorStack();
    }
}

void Mesh::read_boundary_faces(H5::H5File &file) {
    hsize_t dims[2]; // buffer to store size in each dimensions
    nBFG = BFG_names.size();
    nBFace = 0;

    int ibface_global = 0;

    for (string BFG_name: BFG_names) {
        string dset_name = "BFG_" + BFG_name + "_nBFace";
        DataSet dataset = file.openDataSet(dset_name);
        DataSpace dataspace = dataset.getSpace();
        dims[0] = 1;
        int rank = 1;
        DataSpace mspace(rank, dims);
        int nBface_in_group = -1;
        dataset.read(&nBface_in_group, PredType::NATIVE_INT, mspace, dataspace);
        nBFace += nBface_in_group;

        BFG_to_nBFace[BFG_name] = nBface_in_group;
        BFG_to_data[BFG_name].resize(nBface_in_group);
        dims[0] = nBface_in_group;
        dims[1] = 3;
        rank = 2;
        mspace = DataSpace(rank, dims);
        dset_name = "BFG_" + BFG_name + "_BFaceData";
        dataset = file.openDataSet(dset_name);
        dataspace = dataset.getSpace();
        vector<int> buff(dims[0]*dims[1], 0);
        dataset.read(buff.data(), PredType::NATIVE_INT, mspace, dataspace);
        for (int i=0; i<nBface_in_group; i++) {
            BFG_to_data[BFG_name][i].resize(3);
            BFG_to_data[BFG_name][i][0] = buff[3*i + 0];
            BFG_to_data[BFG_name][i][1] = buff[3*i + 1];
            BFG_to_data[BFG_name][i][2] = buff[3*i + 2];

            // Set reverse mapping
            elem_to_BFace[buff[3*i + 0]][elem_num_BFace[buff[3*i + 0]]] = ibface_global;
            elem_num_BFace[buff[3*i + 0]]++;
            ibface_global++;
        }
    }
}

void Mesh::partition(int nparts) {
    idx_t objval;
    idx_t ncommon = 1;
    int ierr = METIS_PartMeshDual(&nElem, &nNode,
        eptr.data(), eind.data(),
        NULL, NULL, &ncommon, &nparts, NULL, NULL, &objval,
        elem_part_id.data(),
        node_part_id.data());

    if (ierr != METIS_OK) cout << "Error partitioning the mesh the mesh." << endl;
    this->nPart =  nparts;
    this->partitioned = true;
}

ostream& operator<<(ostream& os, const Mesh& mesh) {
    os << endl << string(80, '=') << endl;
    os << "---> Mesh info" << endl;
    os << "nElem  = " << mesh.nElem << endl
       << "nNode  = " << mesh.nNode << endl
       << "nIface = " << mesh.nIface << endl
       << "order  = " << mesh.order << endl;
    if (mesh.nBFG>0) {
        os << "--> Mesh has boundaries" << endl;
        os << "nBFG   = " << mesh.nBFG << ", nBFace = " << mesh.nBFace << endl;
        os << "Groups: ";
        for (string name: mesh.BFG_names) {
            os << name << " ";
        }
        os << endl;
    }
    else {
        os << "--> No boundary group. Assuming everything is periodic." << endl;
    }
    if (mesh.partitioned) {
        os << "--> Mesh is partitioned: " << endl;
        for (int part=0; part<mesh.nPart; part++) {
            int nElem_in_part = count(mesh.elem_part_id.begin(), mesh.elem_part_id.end(), part);
            os << "Elements in partition " << part << ": " << nElem_in_part << endl;
        }
    }
    else {
        os << "--> Mesh is not partitioned." << endl;
    }
    os << string(80, '=') << endl << endl;

    return os;
}