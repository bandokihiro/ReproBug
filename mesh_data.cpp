//
// Created by kihiro on 1/28/20.
//

#include <bitset>
#include <iostream>
#include <numeric>
#include "legion.h"
#include "mesh_data.h"
#include "mesh.h"
#include "typedefs.h"

#define N_SUBREGIONS_MAX 200

using namespace Legion;
using namespace LegionRuntime;
using namespace std;

MeshData::MeshData(Context ctx, HighLevelRuntime *runtime, Legion::Logger &logger) :
    LegionData(ctx, runtime, logger), nPart(-1) {}

void MeshData::clean_up() {
    runtime->destroy_index_partition(ctx, elem_lp.get_index_partition());
    runtime->destroy_index_partition(ctx, elem_with_halo_lp.get_index_partition());
    runtime->destroy_index_partition(ctx, iface_lp.get_index_partition());
    runtime->destroy_index_partition(ctx, iface_all_lp.get_index_partition());

    runtime->destroy_logical_partition(ctx, elem_lp);
    runtime->destroy_logical_partition(ctx, elem_with_halo_lp);
    runtime->destroy_logical_partition(ctx, iface_lp);
    runtime->destroy_logical_partition(ctx, iface_all_lp);

    runtime->destroy_field_space(ctx, elem_lr.get_field_space());
    runtime->destroy_field_space(ctx, iface_lr.get_field_space());

    runtime->destroy_index_space(ctx, elem_lr.get_index_space());
    runtime->destroy_index_space(ctx, iface_lr.get_index_space());

    runtime->destroy_logical_region(ctx, elem_lr);
    runtime->destroy_logical_region(ctx, iface_lr);
}

void MeshData::init_mesh_region_elem(const Mesh &mesh) {
    nElem = mesh.nElem;
    nPart = mesh.nPart;
    nIface = mesh.nIface;

    // create index space
    Rect<1> rect(0, nElem - 1);
    IndexSpace is = runtime->create_index_space(ctx, rect);
    runtime->attach_name(is, "mesh_elem_index_space");

    // create field space and allocate
    FieldSpace fs = runtime->create_field_space(ctx);
    runtime->attach_name(fs, "mesh_elem_field_space");

    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(Point<1>), FID_MESH_ELEM_PARTID);
    allocator.allocate_field(sizeof(bitset<N_SUBREGIONS_MAX>), FID_MESH_ELEM_GHOST_BITMASK);

    runtime->attach_name(fs, FID_MESH_ELEM_PARTID, "mesh_elem_partition_id");
    runtime->attach_name(fs, FID_MESH_ELEM_GHOST_BITMASK, "mesh_elem_ghost_bitmask_id");

    // create logical region
    elem_lr = runtime->create_logical_region(ctx, is, fs);
    runtime->attach_name(elem_lr, "mesh_elem_logical_region");

    // define region requirement that determines what to map as well as privileges
    RegionRequirement req(elem_lr, WRITE_DISCARD, EXCLUSIVE, elem_lr);
    req.add_field(FID_MESH_ELEM_PARTID);
    req.add_field(FID_MESH_ELEM_GHOST_BITMASK);

    InlineLauncher inline_launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, inline_launcher);
    pr.wait_until_valid(true);

    const AccWDPoint1 acc_partid(pr, FID_MESH_ELEM_PARTID, sizeof(Point<1>), false, true);
    const FieldAccessor< WRITE_DISCARD, bitset<N_SUBREGIONS_MAX>, 1, coord_t,
        Realm::AffineAccessor<bitset<N_SUBREGIONS_MAX>, 1, coord_t> >
            acc_ghost_bitmask(pr, FID_MESH_ELEM_GHOST_BITMASK, sizeof(bitset<N_SUBREGIONS_MAX>), false, true);

    // initialize
    int ielem = 0;
    // loop over elements
    for (PointInRectIterator<1> pir(rect); pir(); pir++) {
        acc_partid[*pir] = mesh.elem_part_id[ielem];
        acc_ghost_bitmask[*pir].reset();
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
    allocator.allocate_field(sizeof(ElemType), FID_MESH_IFACE_ELEMLTYPE);
    allocator.allocate_field(sizeof(ElemType), FID_MESH_IFACE_ELEMRTYPE);

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
    pr.wait_until_valid(true);

    // accessors
    AccWDPoint1  acc_point[2];
    acc_point[0] = AccWDPoint1(pr, FID_MESH_IFACE_ELEMLID, sizeof(Point<1>), false, true);
    acc_point[1] = AccWDPoint1(pr, FID_MESH_IFACE_ELEMRID, sizeof(Point<1>), false, true);

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

void MeshData::partition_mesh_region() {
    // partition elements
    IndexSpace part_is = runtime->create_index_space(ctx, Rect<1>(0, nPart-1));
    runtime->attach_name(part_is, "mesh_elem_partition_index_space");
    IndexPartition elem_ip = runtime->create_partition_by_field(ctx,
                                                                elem_lr, elem_lr, FID_MESH_ELEM_PARTID, part_is);
    runtime->attach_name(elem_ip, "mesh_elem_index_partition");
    elem_lp = runtime->get_logical_partition(ctx, elem_lr, elem_ip);
    runtime->attach_name(elem_lp, "mesh_elem_logical_partition");
    domain = Domain::from_rect<1>(Arrays::Rect<1>(Arrays::Point<1>(0),  Arrays::Point<1>(nPart-1)));

    // partition interior faces
    IndexPartition iface_ipL = runtime->create_partition_by_preimage(ctx, elem_ip,
            iface_lr, iface_lr, FID_MESH_IFACE_ELEMLID, part_is, DISJOINT_COMPLETE_KIND);
    runtime->attach_name(iface_ipL, "mesh_iface_index_partition_preimage_left");
    IndexPartition iface_ipR = runtime->create_partition_by_preimage(ctx, elem_ip,
            iface_lr, iface_lr, FID_MESH_IFACE_ELEMRID, part_is, DISJOINT_COMPLETE_KIND);
    runtime->attach_name(iface_ipR, "mesh_iface_index_partition_preimage_right");
    IndexPartition iface_ip = runtime->create_partition_by_union(ctx, iface_lr.get_index_space(),
            iface_ipL, iface_ipR, part_is, ALIASED_COMPLETE_KIND);
    runtime->attach_name(iface_ip, "mesh_iface_index_partition_union_left_right");
    iface_lp = runtime->get_logical_partition(ctx, iface_lr, iface_ipL);
    runtime->attach_name(iface_lp, "mesh_iface_logical_parittion_premiage_left");
    iface_all_lp = runtime->get_logical_partition(ctx, iface_lr, iface_ip);
    runtime->attach_name(iface_all_lp, "mesh_iface_logical_partition_union_left_right");

    // partition element with halo
    IndexPartition ip1 = runtime->create_partition_by_image(ctx, elem_lr.get_index_space(),
            iface_all_lp, iface_lr, FID_MESH_IFACE_ELEMLID, part_is, ALIASED_KIND);
    runtime->attach_name(ip1, "mesh_elem_index_partition_image_left");
    IndexPartition ip2 = runtime->create_partition_by_image(ctx, elem_lr.get_index_space(),
            iface_all_lp, iface_lr, FID_MESH_IFACE_ELEMRID, part_is, ALIASED_KIND);
    runtime->attach_name(ip2, "mesh_elem_index_partition_image_right");
    IndexPartition ip = runtime->create_partition_by_union(ctx, elem_lr.get_index_space(),
            ip1, ip2, part_is, ALIASED_COMPLETE_KIND);
    runtime->attach_name(ip, "mesh_elem_with_halo_index_partition");
    elem_with_halo_lp = runtime->get_logical_partition(ctx, elem_lr, ip);
    runtime->attach_name(elem_with_halo_lp, "mesh_elem_with_halo_logical_partition");

    // global index spaces for private and shared elements
    IndexPartition ghost_ip = runtime->create_partition_by_difference(ctx,
            elem_lr.get_index_space(), elem_with_halo_lp.get_index_partition(),
            elem_lp.get_index_partition(), part_is, ALIASED_INCOMPLETE_KIND);
    runtime->attach_name(ghost_ip, "mesh_ghost_elem_all_index_partition");
    IndexSpace privacy_is = runtime->create_index_space(ctx, Rect<1>(0, 1));
    runtime->attach_name(privacy_is, "mesh_privacy_index_space");
    IndexPartition privacy_ip = runtime->create_pending_partition(ctx, elem_lr.get_index_space(),
            privacy_is, DISJOINT_COMPLETE_KIND);
    runtime->attach_name(privacy_ip, "mesh_privacy_index_partition");
    all_shared_is = runtime->create_index_space_union(ctx, privacy_ip, 1, ghost_ip);
    runtime->attach_name(all_shared_is, "mesh_shared_elem_index_space");
    vector<IndexSpace> diff_spaces(1, all_shared_is);
    IndexSpace all_private_is = runtime->create_index_space_difference(ctx, privacy_ip, 0,
            elem_lr.get_index_space(), diff_spaces);
    runtime->attach_name(all_private_is, "mesh_private_elem_index_space");

    // partition of private and shared elements
    map<IndexSpace,IndexPartition> partition_handles;
    partition_handles[all_private_is] = IndexPartition::NO_PART;
    partition_handles[all_shared_is] = IndexPartition::NO_PART;
    runtime->create_cross_product_partitions(ctx, privacy_ip, elem_lp.get_index_partition(),
            partition_handles, DISJOINT_COMPLETE_KIND);
    assert(partition_handles[all_private_is].exists());
    assert(partition_handles[all_shared_is].exists());
    priv_elem_ip = partition_handles[all_private_is];
    runtime->attach_name(priv_elem_ip, "mesh_private_elem_only_index_partition");
    shared_elem_ip = partition_handles[all_shared_is];
    runtime->attach_name(shared_elem_ip, "mesh_shared_elem_only_index_partition");
    ghost_elem_ip = runtime->create_partition_by_difference(ctx, all_shared_is,
            elem_with_halo_lp.get_index_partition(), elem_lp.get_index_partition(), part_is,
            ALIASED_COMPLETE_KIND);
    runtime->attach_name(ghost_elem_ip, "mesh_ghost_elem_based_on_shared_only_index_partition");

    logger.print("Mesh region successfully partitioned");
}

void MeshData::reinit_mesh_region(const Mesh &mesh) {
    /* The first step consists of rearranging the element indexing in order to minimize the
     * number of reduction instances needed for interior faces residual. For this, we sort
     * elements depending on their ghost element ID. An ID of -1 means it is a private element.
     * An ID >= 0 means that this element is shared with sub-region ID (note that this ID is not
     * unique and currently the largest ID is registered by a sequence of fill operations).
     */

    // fill with ghost IDs
    {
        LogicalPartition ghost_elem_lp = runtime->get_logical_partition(ctx, elem_lr,
                ghost_elem_ip);
        for (Domain::DomainPointIterator itr(domain); itr; itr++) {
            Point<1> p = *itr;

            // more sophisticated bitmask method
            LogicalRegion ghost_elem_lsr =
                    runtime->get_logical_subregion_by_color(ctx, ghost_elem_lp, itr.p);
            RegionRequirement req(ghost_elem_lsr, READ_WRITE, EXCLUSIVE, elem_lr);
            req.add_field(FID_MESH_ELEM_GHOST_BITMASK);
            InlineLauncher inline_launcher(req);
            PhysicalRegion pr = runtime->map_region(ctx, inline_launcher);
            pr.wait_until_valid(true /*silence warnings*/);

            const FieldAccessor< READ_WRITE, bitset<N_SUBREGIONS_MAX>, 1, coord_t,
                Realm::AffineAccessor<bitset<N_SUBREGIONS_MAX>, 1, coord_t> >
                    acc_ghost_bitmask(pr, FID_MESH_ELEM_GHOST_BITMASK, sizeof(bitset<N_SUBREGIONS_MAX>), false, true);
            Domain d = runtime->get_index_space_domain(ctx, ghost_elem_lsr.get_index_space());
            for (Domain::DomainPointIterator i(d); i; i++) {
                acc_ghost_bitmask[*i][p[0]] = true;
            }

            runtime->unmap_region(ctx, pr);
        }
    }

    vector<int> new_elem_idx(nElem);
    iota(new_elem_idx.begin(), new_elem_idx.end(), 0);
    vector<int> old_to_new_elem_idx(nElem);
    iota(old_to_new_elem_idx.begin(), old_to_new_elem_idx.end(), 0);

    {
        RegionRequirement req(elem_lr, READ_ONLY, EXCLUSIVE, elem_lr);
        req.add_field(FID_MESH_ELEM_GHOST_BITMASK);
        InlineLauncher inline_launcher(req);
        PhysicalRegion pr = runtime->map_region(ctx, inline_launcher);
        pr.wait_until_valid(true /*silence warnings*/);

        // more sophisticated bitmask method
        const FieldAccessor< READ_ONLY, bitset<N_SUBREGIONS_MAX>, 1, coord_t,
                Realm::AffineAccessor<bitset<N_SUBREGIONS_MAX>, 1, coord_t> >
                acc_ghost_bitmask(pr, FID_MESH_ELEM_GHOST_BITMASK, sizeof(bitset<N_SUBREGIONS_MAX>), false, true);
        stable_sort(new_elem_idx.begin(), new_elem_idx.end(),
                [&mesh, &acc_ghost_bitmask](int a, int b) {
                        if (mesh.elem_part_id[a]!=mesh.elem_part_id[b]) {
                            return mesh.elem_part_id[a] < mesh.elem_part_id[b];
                        }
                        return acc_ghost_bitmask[Point<1>(a)].to_ulong() <
                               acc_ghost_bitmask[Point<1>(b)].to_ulong() ;
                }
        );

        stable_sort(old_to_new_elem_idx.begin(), old_to_new_elem_idx.end(),
                [&new_elem_idx](int a, int b) { return new_elem_idx[a] < new_elem_idx[b]; });

        runtime->unmap_region(ctx, pr);
    }

    /* The second step re-initializes the mesh element region based on the new indexing.
     * The node coordinates are initialized here.
     */

    {
        // define region requirement that determines what to map as well as privileges
        RegionRequirement req(elem_lr, WRITE_DISCARD, EXCLUSIVE, elem_lr);
        req.add_field(FID_MESH_ELEM_PARTID);

        InlineLauncher inline_launcher(req);
        PhysicalRegion pr = runtime->map_region(ctx, inline_launcher);
        pr.wait_until_valid(true /*silence warnings*/);

        const AccWDPoint1 acc_partid(pr, FID_MESH_ELEM_PARTID, sizeof(Point<1>),
                false /*check field size*/, true /*silence warnings*/);

        // loop over elements
        int ielem = 0;
        Rect<1> rect(0, nElem - 1);
        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            int actual_ielem = new_elem_idx[ielem];
            acc_partid[*pir] = mesh.elem_part_id[actual_ielem];

            ielem += 1;
        }

        runtime->unmap_region(ctx, pr);
    }

    /* The third step re-initializes the mesh interior face region.
     */

    {
        // define region requirement that determines what to map as well as privileges
        RegionRequirement req(iface_lr, WRITE_DISCARD, EXCLUSIVE, iface_lr);
        req.add_field(FID_MESH_IFACE_ELEMLID);
        req.add_field(FID_MESH_IFACE_ELEMRID);

        InlineLauncher inline_launcher(req);
        PhysicalRegion pr = runtime->map_region(ctx, inline_launcher);
        pr.wait_until_valid(true /*silence warnings*/);

        // accessors
        AccWDPoint1 acc_point[2];
        acc_point[0] = AccWDPoint1(pr, FID_MESH_IFACE_ELEMLID, sizeof(Point<1>),
                false /*check field size*/, true /*silence warnings*/);
        acc_point[1] = AccWDPoint1(pr, FID_MESH_IFACE_ELEMRID, sizeof(Point<1>),
                false /*check field size*/, true /*silence warnings*/);

        // loop over elements
        int iface = 0;
        Rect<1> rect(0, nIface - 1);
        for (PointInRectIterator<1> pir(rect); pir(); pir++, iface++) {
            // left and right elements ID as Point<1>
            int actual_elemL = old_to_new_elem_idx[mesh.IFace_to_elem[iface][0]];
            int actual_elemR = old_to_new_elem_idx[mesh.IFace_to_elem[iface][3]];
            acc_point[0][*pir] = actual_elemL;
            acc_point[1][*pir] = actual_elemR;
        }

        runtime->unmap_region(ctx, pr);
    }

    /* Finally we partition using the same method but based on the new indexing.
     * All required partitions should be computed here and SolutionData should use the relevant
     * IndexPartition objects instead of calling Legion routines to compute them.
     */

    partition_mesh_region();

    // fill with ghost IDs
    {
        // define region requirement that determines what to map as well as privileges
        RegionRequirement req(elem_lr, WRITE_DISCARD, EXCLUSIVE, elem_lr);
        req.add_field(FID_MESH_ELEM_GHOST_BITMASK);

        InlineLauncher inline_launcher(req);
        PhysicalRegion pr = runtime->map_region(ctx, inline_launcher);
        pr.wait_until_valid(true);

        const FieldAccessor< WRITE_DISCARD, bitset<N_SUBREGIONS_MAX>, 1, coord_t,
                Realm::AffineAccessor<bitset<N_SUBREGIONS_MAX>, 1, coord_t> >
                acc_ghost_bitmask(pr, FID_MESH_ELEM_GHOST_BITMASK, sizeof(bitset<N_SUBREGIONS_MAX>), false, true);

        // initialize
        Rect<1> rect(0, nElem-1);
        // loop over elements
        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            acc_ghost_bitmask[*pir].reset();
        }

        runtime->unmap_region(ctx, pr);
    }
    {
        LogicalPartition ghost_elem_lp = runtime->get_logical_partition(ctx, elem_lr, ghost_elem_ip);
        for (Domain::DomainPointIterator itr(domain); itr; itr++) {
            Point<1> p = *itr;

            // more sophisticated bitmask method
            LogicalRegion ghost_elem_lsr =
                    runtime->get_logical_subregion_by_color(ctx, ghost_elem_lp, itr.p);
            RegionRequirement req(ghost_elem_lsr, READ_WRITE, EXCLUSIVE, elem_lr);
            req.add_field(FID_MESH_ELEM_GHOST_BITMASK);
            InlineLauncher inline_launcher(req);
            PhysicalRegion pr = runtime->map_region(ctx, inline_launcher);
            pr.wait_until_valid(true /*silence warnings*/);

            const FieldAccessor< READ_WRITE, bitset<N_SUBREGIONS_MAX>, 1, coord_t,
                    Realm::AffineAccessor<bitset<N_SUBREGIONS_MAX>, 1, coord_t> >
                    acc_ghost_bitmask(pr, FID_MESH_ELEM_GHOST_BITMASK, sizeof(bitset<N_SUBREGIONS_MAX>), false, true);
            Domain d = runtime->get_index_space_domain(ctx, ghost_elem_lsr.get_index_space());
            for (Domain::DomainPointIterator i(d); i; i++) {
                acc_ghost_bitmask[*i][p[0]] = true;
            }

            runtime->unmap_region(ctx, pr);
        }
    }
}

void MeshData::check_bitmasks_shared() {
    LogicalPartition shared_elem_lp = runtime->get_logical_partition(ctx, elem_lr, shared_elem_ip);
    for (Domain::DomainPointIterator itr(domain); itr; itr++) {
        LogicalRegion shared_elem_lsr =
                runtime->get_logical_subregion_by_color(ctx, shared_elem_lp, itr.p);

        RegionRequirement req(shared_elem_lsr, READ_ONLY, EXCLUSIVE, elem_lr);
        req.add_field(FID_MESH_ELEM_GHOST_BITMASK);

        InlineLauncher inline_launcher(req);
        PhysicalRegion pr = runtime->map_region(ctx, inline_launcher);
        pr.wait_until_valid(true /*silence warnings*/);

        const FieldAccessor< READ_ONLY, bitset<N_SUBREGIONS_MAX>, 1, coord_t,
                Realm::AffineAccessor<bitset<N_SUBREGIONS_MAX>, 1, coord_t> >
                acc_ghost_bitmask(pr, FID_MESH_ELEM_GHOST_BITMASK, sizeof(bitset<N_SUBREGIONS_MAX>), false, true);

        Domain d = runtime->get_index_space_domain(ctx, shared_elem_lsr.get_index_space());
        cout << "Checking shared elements in subregion " << itr.p[0] << endl;
        for (Domain::DomainPointIterator i(d); i; i++) {
            cout << "Element " << i.p[0] << " shared with subregions ";
            for (int j=0; j<nPart; j++) {
                if (acc_ghost_bitmask[*i][j]) {
                    cout << j << " ";
                }
            }
            cout << endl;
        }

        runtime->unmap_region(ctx, pr);
    }
}

void MeshData::check_bitmasks_ghost() {
    LogicalPartition ghost_elem_lp = runtime->get_logical_partition(ctx, elem_lr, ghost_elem_ip);
    for (Domain::DomainPointIterator itr(domain); itr; itr++) {
        LogicalRegion ghost_elem_lsr =
                runtime->get_logical_subregion_by_color(ctx, ghost_elem_lp, itr.p);

        RegionRequirement req(ghost_elem_lsr, READ_ONLY, EXCLUSIVE, elem_lr);
        req.add_field(FID_MESH_ELEM_GHOST_BITMASK);

        InlineLauncher inline_launcher(req);
        PhysicalRegion pr = runtime->map_region(ctx, inline_launcher);
        pr.wait_until_valid(true /*silence warnings*/);

        const FieldAccessor< READ_ONLY, bitset<N_SUBREGIONS_MAX>, 1, coord_t,
                Realm::AffineAccessor<bitset<N_SUBREGIONS_MAX>, 1, coord_t> >
                acc_ghost_bitmask(pr, FID_MESH_ELEM_GHOST_BITMASK, sizeof(bitset<N_SUBREGIONS_MAX>), false, true);

        Domain d = runtime->get_index_space_domain(ctx, ghost_elem_lsr.get_index_space());
        cout << "Checking ghost elements in subregion " << itr.p[0] << endl;
        for (Domain::DomainPointIterator i(d); i; i++) {
            cout << "Element " << i.p[0] << " ghost for subregions ";
            for (int j=0; j<nPart; j++) {
                if (acc_ghost_bitmask[*i][j]) {
                    cout << j << " ";
                }
            }
            cout << endl;
        }

        runtime->unmap_region(ctx, pr);
    }
}