//
// Created by kihiro on 2/2/20.
//

#ifndef DG_TYPEDEFS_H
#define DG_TYPEDEFS_H

#include "legion.h"
#include "types.h"

/*! \brief Read-only accesor for int data
 *
 */
typedef  Legion::FieldAccessor<READ_ONLY, int, 1> AccROint;
/*! \brief Read-write accesor for int data
 *
 */
typedef  Legion::FieldAccessor<READ_WRITE, int, 1> AccRWint;
/*! \brief Write-discard accesor for int data
 *
 */
typedef  Legion::FieldAccessor<WRITE_DISCARD, int, 1> AccWDint;
/*! \brief Reduce accesor for int data
 *
 */
typedef  Legion::FieldAccessor<REDUCE, int, 1> AccREDint;

/*! \brief Read-only accesor for rtype data
 *
 */
typedef  Legion::FieldAccessor<READ_ONLY, rtype, 1> AccROrtype;
/*! \brief Read-write accesor for rtype data
 *
 */
typedef  Legion::FieldAccessor<READ_WRITE, rtype, 1> AccRWrtype;
/*! \brief Write-discard accesor for rtype data
 *
 */
typedef  Legion::FieldAccessor<WRITE_DISCARD, rtype, 1> AccWDrtype;
/*! \brief Reduce accesor for rtype data
 *
 */
typedef  Legion::FieldAccessor<REDUCE, rtype, 1> AccREDrtype;

/*! \brief Read-only accessor for Point<1> data
 *
 */
typedef Legion::FieldAccessor<READ_ONLY, Legion::Point<1>, 1> AccROPoint1;
/*! \brief Write-discard accessor for Point<1> data
 *
 */
typedef Legion::FieldAccessor<WRITE_DISCARD, Legion::Point<1>, 1> AccWDPoint1;

/*! \brief Affine read-only accessor for rtype data
 *
 */
typedef Legion::FieldAccessor< READ_ONLY, rtype, 1, Legion::coord_t,
    Realm::AffineAccessor<rtype, 1, Legion::coord_t> > AffAccROrtype;
/*! \brief Affine read-write accessor for rtype data
 *
 */
typedef Legion::FieldAccessor< READ_WRITE, rtype, 1, Legion::coord_t,
    Realm::AffineAccessor<rtype, 1, Legion::coord_t> > AffAccRWrtype;
/*! \brief Affine read-write accessor for rtype data
 *
 */
typedef Legion::FieldAccessor< WRITE_DISCARD, rtype, 1, Legion::coord_t,
    Realm::AffineAccessor<rtype, 1, Legion::coord_t> > AffAccWDrtype;
/*! \brief Affine reduce accessor for rtype data
 *
 */
typedef Legion::FieldAccessor< REDUCE, rtype, 1, Legion::coord_t,
    Realm::AffineAccessor<rtype, 1, Legion::coord_t> > AffAccREDrtype;

/*! \brief Affine read-write accessor for int data
 *
 */
typedef Legion::FieldAccessor< READ_ONLY, int, 1, Legion::coord_t,
    Realm::AffineAccessor<int, 1, Legion::coord_t> > AffAccROint;
/*! \brief Affine read-write accessor for int data
 *
 */
typedef Legion::FieldAccessor< READ_WRITE, int, 1, Legion::coord_t,
    Realm::AffineAccessor<int, 1, Legion::coord_t> > AffAccRWint;
/*! \brief Affine write-discard accessor for int data
 *
 */
typedef Legion::FieldAccessor< WRITE_DISCARD, int, 1, Legion::coord_t,
    Realm::AffineAccessor<int, 1, Legion::coord_t> > AffAccWDint;
/*! \brief Affine reduce accessor for int data
 *
 */
typedef Legion::FieldAccessor< REDUCE, int, 1, Legion::coord_t,
    Realm::AffineAccessor<int, 1, Legion::coord_t> > AffAccREDint;

/*! \brief Read-only affine accessor for Point<1> data
 *
 */
typedef Legion::FieldAccessor< READ_ONLY, Legion::Point<1>, 1, Legion::coord_t,
    Realm::AffineAccessor<Legion::Point<1>, 1, Legion::coord_t> > AffAccROPoint1;

/*! \brief Affine write-discard accessor for Point<1> data
 *
 */
typedef Legion::FieldAccessor< WRITE_DISCARD, Legion::Point<1>, 1, Legion::coord_t,
    Realm::AffineAccessor<Legion::Point<1>, 1, Legion::coord_t> > AffAccWDPoint1;

#endif //DG_TYPEDEFS_H
