//
// Created by kihiro on 1/13/20.
//

#ifndef DG_TYPES_H
#define DG_TYPES_H

// compiler directive to switch between single and double precision
#ifdef USE_DOUBLES
typedef double rtype;
#else
typedef float rtype;
#endif

#endif //DG_TYPES_H
