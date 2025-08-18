/* Population genetic functions modified for tetraploids
*/

// First, for the autos
double Vfunc_tetra(double x, double nu);
double Mfunc1D_auto(double x, double gam1, double gam2, double gam3, double gam4);
double Mfunc2D_auto(double x, double y, double mxy, 
                    double gam1, double gam2, double gam3, double gam4);
double Mfunc3D_auto(double x, double y, double z, double mxy, double mxz, 
                    double gam1, double gam2, double gam3, double gam4);
double Mfunc4D_auto(double x, double y, double z, double a, 
                    double mxy, double mxz, double mxa, 
                    double gam1, double gam2, double gam3, double gam4);
double Mfunc5D_auto(double x, double y, double z, double a, double b, 
                    double mxy, double mxz, double mxa, double mxb, 
                    double gam1, double gam2, double gam3, double gam4);

// Then for allotetraploids
double Mfunc2D_allo_a(double x, double y, double exy, 
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22);
double Mfunc2D_allo_b(double x, double y, double exy, 
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22);
double Mfunc3D_allo_a(double x, double y, double z, double exy, double mxz, 
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22);
double Mfunc3D_allo_b(double x, double y, double z, double exy, double mxz, 
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22);
double Mfunc4D_allo_a(double x, double y, double z, double a, 
                      double exy, double mxz, double mxa, 
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22);
double Mfunc4D_allo_b(double x, double y, double z, double a, 
                      double exy, double mxz, double mxa, 
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22);
double Mfunc5D_allo_a(double x, double y, double z, double a, double b,
                      double exy, double mxz, double mxa, double mxb, 
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22);
double Mfunc5D_allo_b(double x, double y, double z, double a, double b, 
                      double exy, double mxz, double mxa, double mxb, 
                      double g01, double g02, double g10, double g11, 
                      double g12, double g20, double g21, double g22);

// Autohexaploid
double Vfunc_hex(double x, double nu);
double Mfunc1D_autohex(double x, double g1, double g2, double g3, double g4, double g5, double g6);
double Mfunc2D_autohex(double x, double y, double mxy, double g1, double g2, double g3, double g4, double g5, double g6);
double Mfunc3D_autohex(double x, double y, double z, double mxy, double mxz, 
                   double g1, double g2, double g3, double g4, double g5, double g6);
double Mfunc4D_autohex(double x, double y, double z, double a, double mxy, double mxz, double mxa, 
                   double g1, double g2, double g3, double g4, double g5, double g6);
double Mfunc5D_autohex(double x, double y, double z, double a, double b,
                   double mxy, double mxz, double mxa, double mxb,
                   double g1, double g2, double g3, double g4, double g5, double g6);

// 4+2 hexaploids
double Mfunc2D_hex_tetra(double x, double y, double exy, double g01, double g02, 
                    double g10, double g11, double g12, double g20, double g21, double g22, 
                    double g30, double g31, double g32, double g40, double g41, double g42);
double Mfunc2D_hex_dip(double x, double y, double exy, double g01, double g02, 
                    double g10, double g11, double g12, double g20, double g21, double g22, 
                    double g30, double g31, double g32, double g40, double g41, double g42);
double Mfunc3D_hex_tetra(double x, double y, double z, double exy, double mxz, 
                   double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                   double g30, double g31, double g32, double g40, double g41, double g42);
double Mfunc3D_hex_dip(double x, double y, double z, double exy, double mxz, 
                   double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22,                    
                   double g30, double g31, double g32, double g40, double g41, double g42);
double Mfunc4D_hex_tetra(double x, double y, double z, double a, double exy, double mxz, double mxa, 
                   double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                   double g30, double g31, double g32, double g40, double g41, double g42);
double Mfunc4D_hex_dip(double x, double y, double z, double a, double exy, double mxz, double mxa, 
                   double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22, 
                   double g30, double g31, double g32, double g40, double g41, double g42);
double Mfunc5D_hex_tetra(double x, double y, double z, double a, double b, double exy, double mxz, double mxa, double mxb, 
                   double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22,
                   double g30, double g31, double g32, double g40, double g41, double g42);
double Mfunc5D_hex_dip(double x, double y, double z, double a, double b, double exy, double mxz, double mxa, double mxb, 
                   double g01, double g02, double g10, double g11, double g12, double g20, double g21, double g22,
                   double g30, double g31, double g32, double g40, double g41, double g42);