//
// Created by kihiro on 1/28/20.
//

#include <iostream>
#include "legion.h"
#include "mesh_data.h"
#include "mesh.h"
#include "typedefs.h"

using namespace Legion;
using namespace LegionRuntime;
using namespace std;

MeshData::MeshData(Context ctx, HighLevelRuntime *runtime, Legion::Logger &logger) :
    LegionData(ctx, runtime, logger), nPart(-1) {}

void MeshData::clean_up() {
    runtime->destroy_field_space(ctx, elem_lr.get_field_space());
    runtime->destroy_index_space(ctx, elem_lr.get_index_space());
    runtime->destroy_logical_region(ctx, elem_lr);

    runtime->destroy_field_space(ctx, iface_lr.get_field_space());
    runtime->destroy_index_space(ctx, iface_lr.get_index_space());
    runtime->destroy_logical_region(ctx, iface_lr);
}

void MeshData::init_mesh_region_elem(const Mesh &mesh) {
    // create index space
    nElem = mesh.nElem;
    Rect<1> rect(0, nElem - 1);
    IndexSpace is = runtime->create_index_space(ctx, rect);
    runtime->attach_name(is, "mesh_elem_index_space");

    // create field space and allocate
    FieldSpace fs = runtime->create_field_space(ctx);
    runtime->attach_name(fs, "mesh_elem_field_space");
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);

    allocator.allocate_field(sizeof(Point<1>), FID_MESH_ELEM_PARTID);
    runtime->attach_name(fs, FID_MESH_ELEM_PARTID, "mesh_elem_partition_id");

    // create logical region
    elem_lr = runtime->create_logical_region(ctx, is, fs);
    runtime->attach_name(elem_lr, "mesh_elem_logical_region");

    // define region requirement that determines what to map as well as privileges
    RegionRequirement req(elem_lr, WRITE_DISCARD, EXCLUSIVE, elem_lr);
    req.add_field(FID_MESH_ELEM_PARTID);

    InlineLauncher inline_launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, inline_launcher);
    pr.wait_until_valid();

    const AccWDPoint1 acc_partid(pr, FID_MESH_ELEM_PARTID);

    // initialize
    int ielem = 0;
    // loop over elements
    for (PointInRectIterator<1> pir(rect); pir(); pir++) {
        acc_partid[*pir] = mesh.elem_part_id[ielem];
        ielem += 1;
    }

    runtime->unmap_region(ctx, pr);
}

void MeshData::init_mesh_region_iFace(const Mesh &mesh) {
    // create index space
    int nIFace = mesh.nIface;
    Rect<1> rect(0, nIFace-1);
    IndexSpace is = runtime->create_index_space(ctx, rect);
    runtime->attach_name(is, "mesh_iFace_index_space");

    // create field space and allocate
    FieldSpace fs = runtime->create_field_space(ctx);
    runtime->attach_name(fs, "mesh_data_field_space");
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(Point<1>), FID_MESH_IFACE_ELEMLID);
    allocator.allocate_field(sizeof(Point<1>), FID_MESH_IFACE_ELEMRID);

    runtime->attach_name(fs, FID_MESH_IFACE_ELEMLID, "mesh_iface_left_element_id");
    runtime->attach_name(fs, FID_MESH_IFACE_ELEMRID, "mesh_iface_right_element_ID");

    // create logical region
    iface_lr = runtime->create_logical_region(ctx, is, fs);
    runtime->attach_name(iface_lr, "mesh_iface_logical_region");

    // define region requirement that determines what to map as well as privileges
    RegionRequirement req(iface_lr, WRITE_DISCARD, EXCLUSIVE, iface_lr);
    req.add_field(FID_MESH_IFACE_ELEMLID);
    req.add_field(FID_MESH_IFACE_ELEMRID);

    InlineLauncher inline_launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, inline_launcher);
    pr.wait_until_valid();

    // accessors
    AccWDPoint1  acc_point[2];
    acc_point[0] = AccWDPoint1(pr, FID_MESH_IFACE_ELEMLID);
    acc_point[1] = AccWDPoint1(pr, FID_MESH_IFACE_ELEMRID);

    // initialize
    int iface = 0;
    // loop over elements
    for (PointInRectIterator<1> pir(rect); pir(); pir++, iface++) {
        // left and right elements ID as Point<1>
        acc_point[0][*pir] = mesh.IFace_to_elem[iface][0];
        acc_point[1][*pir] = mesh.IFace_to_elem[iface][3];
    }

    runtime->unmap_region(ctx, pr);
}

void MeshData::init_mesh_region(const Mesh &mesh) {
    init_mesh_region_elem(mesh);
    init_mesh_region_iFace(mesh);
    runtime->print_once(ctx, stdout, "Mesh region successfully initialized\n");
}

void MeshData::partition_mesh_region(const int nPart_) {
    nPart = nPart_;

    // partition elements
    IndexSpace part_is = runtime->create_index_space(ctx, Rect<1>(0, nPart-1));
    runtime->attach_name(part_is, "partition_index_space");
    IndexPartition elem_ip = runtime->create_partition_by_field(ctx,
        elem_lr, elem_lr, FID_MESH_ELEM_PARTID, part_is);
    runtime->attach_name(elem_ip, "element_index_partition");
    elem_lp = runtime->get_logical_partition(ctx, elem_lr, elem_ip);
    runtime->attach_name(elem_lp, "element_logical_partition");
    domain = Domain::from_rect<1>(Arrays::Rect<1>(Arrays::Point<1>(0),
        Arrays::Point<1>(nPart-1)));

    // partition ifaces
    IndexPartition iface_ipL = runtime->create_partition_by_preimage(ctx, elem_ip,
        iface_lr, iface_lr, FID_MESH_IFACE_ELEMLID, part_is);
    runtime->attach_name(iface_ipL, "left internal face index partition");
    IndexPartition iface_ipR = runtime->create_partition_by_preimage(ctx, elem_ip,
        iface_lr, iface_lr, FID_MESH_IFACE_ELEMRID, part_is);
    runtime->attach_name(iface_ipR, "right_internal_face_index_partition");
    IndexPartition iface_ip = runtime->create_partition_by_union(ctx, iface_lr.get_index_space(),
        iface_ipL, iface_ipR, part_is);
    runtime->attach_name(iface_ip, "all_internal_face_index_partition");
    iface_lp = runtime->get_logical_partition(ctx, iface_lr, iface_ipL);
    runtime->attach_name(iface_lp, "internal_face_logical_partition");
    iface_all_lp = runtime->get_logical_partition(ctx, iface_lr, iface_ip);
    runtime->attach_name(iface_all_lp, "all_internal_face_logical_partition");

    // partition element with halo
    IndexPartition ip1 = runtime->create_partition_by_image(ctx, elem_lr.get_index_space(),
        iface_all_lp, iface_lr, FID_MESH_IFACE_ELEMLID, part_is);
    runtime->attach_name(ip1, "temporary_index_partition_for_left_elements");
    IndexPartition ip2 = runtime->create_partition_by_image(ctx, elem_lr.get_index_space(),
        iface_all_lp, iface_lr, FID_MESH_IFACE_ELEMRID, part_is);
    runtime->attach_name(ip2, "temporary_index_partition_for_right_elements");
    IndexPartition ip = runtime->create_partition_by_union(ctx, elem_lr.get_index_space(),
        ip1, ip2, part_is);
    runtime->attach_name(ip, "index_partition_for_elements_with_halo");
    elem_with_halo_lp = runtime->get_logical_partition(ctx, elem_lr, ip);
    runtime->attach_name(elem_with_halo_lp, "element_with_halo_logical_partition");
}
